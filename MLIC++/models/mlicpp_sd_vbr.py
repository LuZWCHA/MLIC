import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from compressai.models import CompressionModel
from compressai.ops import quantize_ste as ste_round, LowerBound
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.func import update_registered_buffers, get_scale_table
from utils.ckbd import *
from modules.transform import *
from compressai.entropy_models import EntropyBottleneck, EntropyBottleneckVbr


class MLICPlusPlusSDVbr(CompressionModel):
    def __init__(self, config, vr_entbttlnck=None, **kwargs):
        super().__init__(config.N, **kwargs)
        N = config.N
        M = config.M
        context_window = config.context_window
        slice_num = config.slice_num
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M

        self.N = N
        self.M = M
        self.context_window = context_window
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.g_a = AnalysisTransform(N=N, M=M)
        self.h_a = HyperAnalysis(M=M, N=N)

        # reload
        self.g_s = SynthesisTransform(N=N // 4, M=M)
        self.h_s = HyperSynthesis(M=M // 4, N=N)

        M = M // 4
        N = N // 4

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

        self.local_context = nn.ModuleList(
            LocalContext(dim=slice_ch)
            for _ in range(slice_num)
        )

        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )

        # Global Reference for non-anchors
        self.global_inter_context = nn.ModuleList(
            LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
            for i in range(slice_num)
        )
        self.global_intra_context = nn.ModuleList(
            LinearGlobalIntraContext(dim=slice_ch) if i else None
            for i in range(slice_num)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 10, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )

        # Latent Residual Prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )


        ###################################################### VBR modules ####################################################

        # lambdas to use during training
        self.lmbda = [0.0005, 0.0035, 0.0067, 0.025, 0.0483, 0.18]
        self.levels = len(self.lmbda)
        # gain: inverse of quantization step size
        self.Gain = torch.nn.Parameter(
            torch.tensor(
                [0.06556, 0.13944, 0.19293, 0.37268, 0.51801, 1.00000]
            ),
            requires_grad=True,
        )
        # 3-layer NN to get quant offset from Gain and stdev i.e. scales_hat
        Nds = 12
        self.QuantABCD = nn.Sequential(
            nn.Linear(1 + 1, Nds),
            nn.ReLU(),
            nn.Linear(Nds, Nds),
            nn.ReLU(),
            nn.Linear(Nds, 1),
        )
        # flag to indicate whether to use or not to use quantization offsets
        self.no_quantoffset = True
        # use also variable rate hyper prior z (entropy_bottleneck)
        self.vr_entbttlnck = vr_entbttlnck
        if self.vr_entbttlnck:
            self.entropy_bottleneck = EntropyBottleneckVbr(N)
            # 3-layer NN to get quant step size for hyper prior from Gain
            Ndsz = 10
            self.gayn2zqstep = nn.Sequential(
                nn.Linear(1, Ndsz),
                nn.ReLU(),
                nn.Linear(Ndsz, Ndsz),
                nn.ReLU(),
                nn.Linear(Ndsz, 1),
                nn.Softplus(),
            )
            self.lower_bound_zqstep = LowerBound(0.5)

    def mmo_parameters(self):
        share_p = []
        sp_p = []
        
        for i, v in self.named_parameters():
            if "gain" in i:
                sp_p.append(v)
            else:
                share_p.append(v)
        
        return share_p, sp_p

    def _raise_stage_error(self, stage):
        raise ValueError(f"Invalid stage (stage={stage}) parameter for this model.")

    def _get_scale(self, stage, s, inputscale):
        s = max(0, min(s, len(self.Gain) - 1))  # clips to correct range
        if self.training:
            if stage > 1:
                scale = self.Gain[s].detach()
                # scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps # train scale
            else:
                scale = self.Gain[s].detach()
        else:
            if inputscale == 0:
                scale = self.Gain[s].detach()
            else:
                scale = inputscale
        return scale

    def forward(self, x, stage: int = 2, s: int = 1, inputscale=0):
        """
        Using checkerboard context model with mask attention
        which divides y into anchor and non-anchor parts
        non-anchor use anchor as spatial context
        In addition, a channel-wise entropy model is used, too.
        Args:
            x: [B, 3, H, W]
        return:
            x_hat: [B, 3, H, W]
            y_likelihoods: [B, M, H // 16, W // 16]
            z_likelihoods: [B, N, H // 64, W // 64]
            likelihoods: y_likelihoods, z_likelihoods
        """

        scale = self._get_scale(stage, s, inputscale)
        rescale = 1.0 / scale.clone().detach()
        self.update_resolutions(x.size(2) // 16, x.size(3) // 16, device=x.device)
    
        if stage == 1:
            y = self.g_a(x)
            z = self.h_a(y)

            _, z_likelihoods = self.entropy_bottleneck(z)
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = ste_round(z - z_offset) + z_offset

            # Hyper-parameters
            hyper_params = self.h_s(z_hat)
            hyper_scales, hyper_means = hyper_params.chunk(2, 1)

            y_slices = y.chunk(self.slice_num, dim=1)
            y_hat_slices = []
            y_likelihoods = []
            for idx, y_slice in enumerate(y_slices):
                slice_anchor, slice_nonanchor = ckbd_split(y_slice)
                if idx == 0:
                    # Anchor
                    params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # round anchor
                    slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                    # predict residuals cause by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    # local_ctx: [B, H, W, 2 * C]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # merge means and scales of anchor and nonanchor
                    scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                    means_slice = ckbd_merge(means_anchor, means_nonanchor)
                    _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                    # round slice_nonanchor
                    slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                    y_hat_slice = slice_anchor + slice_nonanchor
                    # predict residuals cause by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                    y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(y_hat_slice)
                    y_likelihoods.append(y_slice_likelihoods)

                else:
                    global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                    channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                    # Anchor(Use channel context and hyper params)
                    params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # round anchor
                    slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                    # predict residuals cause by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor(Use spatial context, channel context and hyper params)
                    global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                    # ctx_params: [B, H, W, 2 * C]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # merge means and scales of anchor and nonanchor
                    scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                    means_slice = ckbd_merge(means_anchor, means_nonanchor)
                    _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                    # round slice_nonanchor
                    slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                    y_hat_slice = slice_anchor + slice_nonanchor
                    # predict residuals cause by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                    y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(y_hat_slice)
                    y_likelihoods.append(y_slice_likelihoods)

            y_hat = torch.cat(y_hat_slices, dim=1)
            y_likelihoods = torch.cat(y_likelihoods, dim=1)
            x_hat = self.g_s(y_hat)
        elif stage == 2:
            y = self.g_a(x)
            z = self.h_a(y)
            if not self.vr_entbttlnck:
                z_hat, z_likelihoods = self.entropy_bottleneck(z)

                z_offset = self.entropy_bottleneck._get_medians()
                z_tmp = z - z_offset
                z_hat = ste_round(z_tmp) + z_offset
            else:
                z_qstep = self.gayn2zqstep(1.0 / scale.clone().view(1))
                z_qstep = self.lower_bound_zqstep(z_qstep)
                z_hat, z_likelihoods = self.entropy_bottleneck(
                    z, qs=z_qstep[0], training=None, ste=False
                )  # ste=True)

            hyper_params = self.h_s(z_hat)
            hyper_scales, hyper_means = hyper_params.chunk(2, 1)
            if self.no_quantoffset:
                y_slices = y.chunk(self.slice_num, dim=1)
                y_hat_slices = []
                y_likelihoods = []
                for idx, y_slice in enumerate(y_slices):
                    slice_anchor, slice_nonanchor = ckbd_split(y_slice)
                    if idx == 0:
                        # Anchor
                        params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                        scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                        # split means and scales of anchor
                        scales_anchor = ckbd_anchor(scales_anchor)
                        means_anchor = ckbd_anchor(means_anchor)
                        # round anchor
                        slice_anchor = ste_round((slice_anchor - means_anchor) * scale) * rescale + means_anchor
                        # predict residuals cause by round
                        lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                        slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                        # Non-anchor
                        # local_ctx: [B, H, W, 2 * C]
                        local_ctx = self.local_context[idx](slice_anchor)
                        params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                        scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                        # split means and scales of nonanchor
                        scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                        means_nonanchor = ckbd_nonanchor(means_nonanchor)
                        # merge means and scales of anchor and nonanchor
                        scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                        means_slice = ckbd_merge(means_anchor, means_nonanchor)
                        _, y_slice_likelihoods = self.gaussian_conditional(y_slice * scale, scales_slice * scale, means_slice * scale)
                        # round slice_nonanchor
                        slice_nonanchor = ste_round((slice_nonanchor - means_nonanchor) * scale) * rescale + means_nonanchor
                        y_hat_slice = slice_anchor + slice_nonanchor
                        # predict residuals cause by round
                        lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                        y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                        y_hat_slices.append(y_hat_slice)
                        y_likelihoods.append(y_slice_likelihoods)

                    else:
                        global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                        channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                        # Anchor(Use channel context and hyper params)
                        params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                        scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                        # split means and scales of anchor
                        scales_anchor = ckbd_anchor(scales_anchor)
                        means_anchor = ckbd_anchor(means_anchor)
                        # round anchor
                        slice_anchor = ste_round((slice_anchor - means_anchor) * scale) * rescale + means_anchor
                        # predict residuals cause by round
                        lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                        slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                        # Non-anchor(Use spatial context, channel context and hyper params)
                        global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                        # ctx_params: [B, H, W, 2 * C]
                        local_ctx = self.local_context[idx](slice_anchor)
                        params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                        scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                        # split means and scales of nonanchor
                        scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                        means_nonanchor = ckbd_nonanchor(means_nonanchor)
                        # merge means and scales of anchor and nonanchor
                        scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                        means_slice = ckbd_merge(means_anchor, means_nonanchor)
                        _, y_slice_likelihoods = self.gaussian_conditional(y_slice * scale, scales_slice * scale, means_slice * scale)
                        # round slice_nonanchor
                        slice_nonanchor = ste_round((slice_nonanchor - means_nonanchor) * scale) * rescale + means_nonanchor
                        y_hat_slice = slice_anchor + slice_nonanchor
                        # predict residuals cause by round
                        lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                        y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                        y_hat_slices.append(y_hat_slice)
                        y_likelihoods.append(y_slice_likelihoods)
            else:
                y_slices = y.chunk(self.slice_num, dim=1)
                y_hat_slices = []
                y_likelihoods = []
                for idx, y_slice in enumerate(y_slices):
                    slice_anchor, slice_nonanchor = ckbd_split(y_slice)

                    if idx == 0:
                        # Anchor
                        params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                        scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                        # split means and scales of anchor
                        scales_anchor = ckbd_anchor(scales_anchor)
                        means_anchor = ckbd_anchor(means_anchor)
                        # round anchor
                        # slice_anchor = ste_round((slice_anchor - means_anchor) * scale) * rescale + means_anchor
                        y_zm = slice_anchor - means_anchor
                        y_zm_sc = y_zm * scale
                        signs = torch.sign(y_zm_sc).detach()
                        q_= ste_round(y_zm_sc)
                        q_stdev = self.gaussian_conditional.lower_bound_scale(
                            scales_anchor * scale
                        )

                        stdev_and_gain = torch.cat(
                            (
                                q_stdev.unsqueeze(dim=2),
                                scale.detach().expand(q_stdev.unsqueeze(dim=2).shape),
                            ),
                            dim=2,
                        )
                        q_offsets = (-1) * (self.QuantABCD.forward(stdev_and_gain)).squeeze(
                            dim=2
                        )
                        q_offsets[-0.0001 < q_ < 0.0001] = (
                            0  # must use zero offset for locations quantized to zero
                        )
                        y_q = signs * (q_ + q_offsets)
                        y_q = y_q * rescale + means_anchor
                        slice_anchor = y_q

                        # predict residuals cause by round
                        lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                        slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                        # Non-anchor
                        # local_ctx: [B, H, W, 2 * C]
                        local_ctx = self.local_context[idx](slice_anchor)
                        params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                        scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                        # split means and scales of nonanchor
                        scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                        means_nonanchor = ckbd_nonanchor(means_nonanchor)
                        # merge means and scales of anchor and nonanchor
                        scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                        means_slice = ckbd_merge(means_anchor, means_nonanchor)
                        _, y_slice_likelihoods = self.gaussian_conditional(y_slice * scale, scales_slice * scale, means_slice)
                        # round slice_nonanchor
                        # slice_nonanchor = ste_round((slice_nonanchor - means_nonanchor) * scale) * rescale + means_nonanchor
                        y_zm = slice_nonanchor - means_nonanchor
                        y_zm_sc = y_zm * scale
                        signs = torch.sign(y_zm_sc).detach()
                        q_= ste_round(y_zm_sc)
                        q_stdev = self.gaussian_conditional.lower_bound_scale(
                            scales_anchor * scale
                        )

                        stdev_and_gain = torch.cat(
                            (
                                q_stdev.unsqueeze(dim=2),
                                scale.detach().expand(q_stdev.unsqueeze(dim=2).shape),
                            ),
                            dim=2,
                        )
                        q_offsets = (-1) * (self.QuantABCD.forward(stdev_and_gain)).squeeze(
                            dim=2
                        )
                        q_offsets[-0.0001 < q_ < 0.0001] = (
                            0  # must use zero offset for locations quantized to zero
                        )
                        y_q = signs * (q_ + q_offsets)
                        y_q = y_q * rescale + means_nonanchor                        
                        slice_nonanchor = y_q

                        y_hat_slice = slice_anchor + slice_nonanchor
                        # predict residuals cause by round
                        lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                        y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                        y_hat_slices.append(y_hat_slice)
                        y_likelihoods.append(y_slice_likelihoods)

                    else:
                        global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                        channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                        # Anchor(Use channel context and hyper params)
                        params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                        scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                        # split means and scales of anchor
                        scales_anchor = ckbd_anchor(scales_anchor)
                        means_anchor = ckbd_anchor(means_anchor)
                        # round anchor
                        # slice_anchor = ste_round((slice_anchor - means_anchor) * scale) * rescale + means_anchor
                        y_zm = slice_anchor - means_anchor
                        y_zm_sc = y_zm * scale
                        signs = torch.sign(y_zm_sc).detach()
                        q_= ste_round(y_zm_sc)
                        q_stdev = self.gaussian_conditional.lower_bound_scale(
                            scales_anchor * scale
                        )

                        stdev_and_gain = torch.cat(
                            (
                                q_stdev.unsqueeze(dim=2),
                                scale.detach().expand(q_stdev.unsqueeze(dim=2).shape),
                            ),
                            dim=2,
                        )
                        q_offsets = (-1) * (self.QuantABCD.forward(stdev_and_gain)).squeeze(
                            dim=2
                        )
                        q_offsets[-0.0001 < q_ < 0.0001] = (
                            0  # must use zero offset for locations quantized to zero
                        )
                        y_q = signs * (q_ + q_offsets)
                        y_q = y_q * rescale + means_anchor
                        slice_anchor = y_q
                        # predict residuals cause by round
                        lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                        slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                        # Non-anchor(Use spatial context, channel context and hyper params)
                        global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                        # ctx_params: [B, H, W, 2 * C]
                        local_ctx = self.local_context[idx](slice_anchor)
                        params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                        scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                        # split means and scales of nonanchor
                        scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                        means_nonanchor = ckbd_nonanchor(means_nonanchor)
                        # merge means and scales of anchor and nonanchor
                        scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                        means_slice = ckbd_merge(means_anchor, means_nonanchor)
                        _, y_slice_likelihoods = self.gaussian_conditional(y_slice * scale, scales_slice * scale, means_slice * scale)
                        # round slice_nonanchor
                        # slice_nonanchor = ste_round((slice_nonanchor - means_nonanchor) * scale) * rescale + means_nonanchor
                        y_zm = slice_nonanchor - means_nonanchor
                        y_zm_sc = y_zm * scale
                        signs = torch.sign(y_zm_sc).detach()
                        q_= ste_round(y_zm_sc)
                        q_stdev = self.gaussian_conditional.lower_bound_scale(
                            scales_anchor * scale
                        )

                        stdev_and_gain = torch.cat(
                            (
                                q_stdev.unsqueeze(dim=2),
                                scale.detach().expand(q_stdev.unsqueeze(dim=2).shape),
                            ),
                            dim=2,
                        )
                        q_offsets = (-1) * (self.QuantABCD.forward(stdev_and_gain)).squeeze(
                            dim=2
                        )
                        q_offsets[-0.0001 < q_ < 0.0001] = (
                            0  # must use zero offset for locations quantized to zero
                        )
                        y_q = signs * (q_ + q_offsets)
                        y_q = y_q * rescale + means_nonanchor                        
                        slice_nonanchor = y_q
                        y_hat_slice = slice_anchor + slice_nonanchor
                        # predict residuals cause by round
                        lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                        y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                        y_hat_slices.append(y_hat_slice)
                        y_likelihoods.append(y_slice_likelihoods)
            
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        }

    def update_resolutions(self, H, W, device=None):
        if device is None:
            for param in self.g_a.parameters():
                if param.device is not None:
                    device = param.device
                # print(device)
                break
        for i in range(len(self.global_intra_context)):
            if i == 0:
                self.local_context[i].update_resolution(H, W, device, mask=None)
            else:
                self.local_context[i].update_resolution(H, W, device, mask=self.local_context[0].attn_mask)


    def compress(self, x, stage: int = 2, s: int = 1, inputscale=0):
        start_time = time.time()
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(
                0, self.levels
            ), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])
        rescale = torch.tensor(1.0) / scale

        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)

        if stage == 1 or (stage == 2 and not self.vr_entbttlnck):
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        elif stage == 2:  # support vr EntropyBottleneck
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_strings = self.entropy_bottleneck.compress(z, qs=z_qstep[0])
            z_hat = self.entropy_bottleneck.decompress(
                z_strings, z.size()[-2:], qs=z_qstep[0]
            )
        else:
            self._raise_stage_error(self, stage)

        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []


        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if stage == 1:
                if idx == 0:
                    # Anchor
                    params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # round and compress anchor
                    slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # round and compress nonanchor
                    slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)
                else:
                    # Anchor
                    global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                    channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                    params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # round and compress anchor
                    slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # round and compress nonanchor
                    slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)
            elif stage == 2:
                if idx == 0:
                    # Anchor
                    params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # round and compress anchor
                    slice_anchor = compress_anchor_vbr(self.gaussian_conditional, self.QuantABCD, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list, scale, rescale, self.no_quantoffset)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # round and compress nonanchor
                    slice_nonanchor = compress_nonanchor_vbr(self.gaussian_conditional, self.QuantABCD, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list, scale, rescale, self.no_quantoffset)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)
                else:
                    # Anchor
                    global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                    channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                    params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # round and compress anchor
                    slice_anchor = compress_anchor_vbr(self.gaussian_conditional, self.QuantABCD, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list, scale, rescale, self.no_quantoffset)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # round and compress nonanchor
                    slice_nonanchor = compress_nonanchor_vbr(self.gaussian_conditional, self.QuantABCD, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list, scale, rescale, self.no_quantoffset)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time
        }



    # def compress(self, x):
    #     torch.cuda.synchronize()
    #     start_time = time.time()
    #     self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
    #     y = self.g_a(x)
    #     z = self.h_a(y)
    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    #     hyper_params = self.h_s(z_hat)
    #     hyper_scales, hyper_means = hyper_params.chunk(2, 1)
    #     y_slices = y.chunk(self.slice_num, dim=1)
    #     y_hat_slices = []

    #     cdf = self.gaussian_conditional.quantized_cdf.tolist()
    #     cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    #     offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
    #     encoder = BufferedRansEncoder()
    #     symbols_list = []
    #     indexes_list = []
    #     y_strings = []

    #     for idx, y_slice in enumerate(y_slices):
    #         slice_anchor, slice_nonanchor = ckbd_split(y_slice)
    #         if idx == 0:
    #             # Anchor
    #             params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
    #             scales_anchor, means_anchor = params_anchor.chunk(2, 1)
    #             # split means and scales of anchor
    #             scales_anchor = ckbd_anchor(scales_anchor)
    #             means_anchor = ckbd_anchor(means_anchor)
    #             # round and compress anchor
    #             slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
    #             # predict residuals caused by round
    #             lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
    #             slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
    #             # Non-anchor
    #             # local_ctx: [B,2 * C, H, W]
    #             local_ctx = self.local_context[idx](slice_anchor)
    #             params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
    #             scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
    #             # split means and scales of nonanchor
    #             scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
    #             means_nonanchor = ckbd_nonanchor(means_nonanchor)
    #             # round and compress nonanchor
    #             slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
    #             # predict residuals caused by round
    #             lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
    #             slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
    #             y_hat_slices.append(slice_nonanchor + slice_anchor)

    #         else:
    #             # Anchor
    #             global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
    #             channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
    #             params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
    #             scales_anchor, means_anchor = params_anchor.chunk(2, 1)
    #             # split means and scales of anchor
    #             scales_anchor = ckbd_anchor(scales_anchor)
    #             means_anchor = ckbd_anchor(means_anchor)
    #             # round and compress anchor
    #             slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
    #             # predict residuals caused by round
    #             lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
    #             slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
    #             # Non-anchor
    #             global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
    #             # local_ctx: [B,2 * C, H, W]
    #             local_ctx = self.local_context[idx](slice_anchor)
    #             params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
    #             scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
    #             # split means and scales of nonanchor
    #             scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
    #             means_nonanchor = ckbd_nonanchor(means_nonanchor)
    #             # round and compress nonanchor
    #             slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
    #             # predict residuals caused by round
    #             lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
    #             slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
    #             y_hat_slices.append(slice_nonanchor + slice_anchor)

    #     encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
    #     y_string = encoder.flush()
    #     y_strings.append(y_string)
    #     torch.cuda.synchronize()
    #     end_time = time.time()

    #     cost_time = end_time - start_time
    #     return {
    #         "strings": [y_strings, z_strings],
    #         "shape": z.size()[-2:],
    #         "cost_time": cost_time
    #     }

    # def decompress(self, strings, shape):
    #     torch.cuda.synchronize()
    #     start_time = time.time()
    #     y_strings = strings[0][0]
    #     z_strings = strings[1]
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
    #     self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
    #     hyper_params = self.h_s(z_hat)
    #     hyper_scales, hyper_means = hyper_params.chunk(2, 1)
    #     y_hat_slices = []

    #     cdf = self.gaussian_conditional.quantized_cdf.tolist()
    #     cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    #     offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
    #     decoder = RansDecoder()
    #     decoder.set_stream(y_strings)

    #     for idx in range(self.slice_num):
    #         if idx == 0:
    #             # Anchor
    #             params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
    #             scales_anchor, means_anchor = params_anchor.chunk(2, 1)
    #             # split means and scales of anchor
    #             scales_anchor = ckbd_anchor(scales_anchor)
    #             means_anchor = ckbd_anchor(means_anchor)
    #             # decompress anchor
    #             slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
    #             # predict residuals caused by round
    #             lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
    #             slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
    #             # Non-anchor
    #             # local_ctx: [B,2 * C, H, W]
    #             local_ctx = self.local_context[idx](slice_anchor)
    #             params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
    #             scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
    #             # split means and scales of nonanchor
    #             scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
    #             means_nonanchor = ckbd_nonanchor(means_nonanchor)
    #             # decompress non-anchor
    #             slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
    #             # predict residuals caused by round
    #             lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
    #             slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
    #             y_hat_slices.append(slice_nonanchor + slice_anchor)

    #         else:
    #             # Anchor
    #             global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
    #             channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
    #             params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
    #             scales_anchor, means_anchor = params_anchor.chunk(2, 1)
    #             # split means and scales of anchor
    #             scales_anchor = ckbd_anchor(scales_anchor)
    #             means_anchor = ckbd_anchor(means_anchor)
    #             # decompress anchor
    #             slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
    #             # predict residuals caused by round
    #             lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
    #             slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
    #             # Non-anchor
    #             # Non-anchor
    #             global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
    #             # local_ctx: [B,2 * C, H, W]
    #             local_ctx = self.local_context[idx](slice_anchor)
    #             params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
    #             scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
    #             # split means and scales of nonanchor
    #             scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
    #             means_nonanchor = ckbd_nonanchor(means_nonanchor)
    #             # decompress non-anchor
    #             slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
    #             # predict residuals caused by round
    #             lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
    #             slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
    #             y_hat_slices.append(slice_nonanchor + slice_anchor)

    #     y_hat = torch.cat(y_hat_slices, dim=1)
    #     x_hat = self.g_s(y_hat)
    #     torch.cuda.synchronize()
    #     end_time = time.time()

    #     cost_time = end_time - start_time

    #     return {
    #         "x_hat": x_hat,
    #         "cost_time": cost_time
    #     }
    
    def decompress(self, strings, shape, stage=2, s=1, inputscale=0):
        torch.cuda.synchronize()
        start_time = time.time()

        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(
                0, self.levels
            ), f"s should in range(0, {self.levels}), but get s:{s}"
            scale = torch.abs(self.Gain[s])

        rescale = torch.tensor(1.0) / scale

        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):

            if stage == 1:
                if idx == 0:
                    # Anchor
                    params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # decompress anchor
                    slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # decompress non-anchor
                    slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)

                else:
                    # Anchor
                    global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                    channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                    params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # decompress anchor
                    slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    # Non-anchor
                    global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # decompress non-anchor
                    slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)
            elif stage == 2:
                if idx == 0:
                    # Anchor
                    params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # decompress anchor
                    slice_anchor = decompress_anchor_vbr(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # decompress non-anchor
                    slice_nonanchor = decompress_nonanchor_vbr(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)

                else:
                    # Anchor
                    global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                    channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                    params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                    # split means and scales of anchor
                    scales_anchor = ckbd_anchor(scales_anchor)
                    means_anchor = ckbd_anchor(means_anchor)
                    # decompress anchor
                    slice_anchor = decompress_anchor_vbr(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                    # predict residuals caused by round
                    lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                    slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                    # Non-anchor
                    # Non-anchor
                    global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                    # local_ctx: [B,2 * C, H, W]
                    local_ctx = self.local_context[idx](slice_anchor)
                    params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                    scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                    # split means and scales of nonanchor
                    scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                    means_nonanchor = ckbd_nonanchor(means_nonanchor)
                    # decompress non-anchor
                    slice_nonanchor = decompress_nonanchor_vbr(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets, scale, rescale)
                    # predict residuals caused by round
                    lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                    slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                    y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def net_decoder_forward(self, x):
        B, C, H, W = x.shape

        z_hat = torch.zeros((B, self.N, H // 64, W // 64), device=x.device)

        self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        # cdf = self.gaussian_conditional.quantized_cdf.tolist()
        # cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        # offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        # decoder = RansDecoder()
        # decoder.set_stream("dasdadsdadasdad")

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = means_anchor.clone()
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = means_anchor.clone()
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = means_anchor.clone()
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = means_anchor.clone()
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        torch.cuda.synchronize()

        return x_hat

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=False)

    # def update(self, scale_table=None, force=False):
    #     if scale_table is None:
    #         scale_table = get_scale_table()
    #     updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
    #     updated |= super().update(force=force)
    #     return updated
    
    @classmethod
    def from_state_dict(cls, state_dict, vr_entbttlnck=False):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M, vr_entbttlnck)
        if "QuantOffset" in state_dict.keys():
            del state_dict["QuantOffset"]
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False, scale=None):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        # update vr EntropyBottleneck with given scale, i.e. quantization step size
        if isinstance(self.entropy_bottleneck, EntropyBottleneckVbr):
            sc = scale
            if sc is None:
                rv = self.entropy_bottleneck.update(force=force)
            else:
                z_qstep = self.gayn2zqstep(1.0 / sc.view(1))
                z_qstep = self.lower_bound_zqstep(z_qstep)
                rv = self.entropy_bottleneck.update_variable(force=force, qs=z_qstep)
        elif isinstance(self.entropy_bottleneck, EntropyBottleneck):
            rv = self.entropy_bottleneck.update(force=force)
        updated |= rv
        return updated