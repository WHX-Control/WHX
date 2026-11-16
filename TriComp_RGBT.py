import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, GhostConv
from .block import C3Ghost

def _gcd_group(c1, c2):
    return max(math.gcd(int(c1), int(c2)), 1)

class OmniLiteChannelGate(nn.Module):

    def __init__(self, c, reduction=8):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, hidden, 1, bias=False), nn.SiLU(inplace=True), nn.Conv2d(hidden, c, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

class OmniLiteSpatialGate(nn.Module):

    def __init__(self, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 1, k, 1, k // 2, bias=False), nn.Sigmoid())

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return self.net(torch.cat([avg, mx], dim=1))

class OmniLiteGhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, reduction=8):
        super().__init__()
        c_primary = max(c2 // 2, 1)
        c_ghost = c2 - c_primary
        self.primary = Conv(c1, c_primary, k, s, g=g, act=act)
        c_g3 = c_ghost // 2
        c_g5 = c_ghost - c_g3
        self.cheap3 = Conv(c_primary, c_g3, 3, 1, g=_gcd_group(c_primary, c_g3), act=act) if c_g3 > 0 else nn.Identity()
        self.cheap5 = Conv(c_primary, c_g5, 5, 1, g=_gcd_group(c_primary, c_g5), act=act) if c_g5 > 0 else nn.Identity()
        self.c2 = c2
        hidden = max(c2 // reduction, 8)
        self.remap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, hidden, 1, bias=False), nn.SiLU(inplace=True), nn.Conv2d(hidden, c2 * 2, 1, bias=True))
        self.sparse_score = nn.Sequential(Conv(c2, hidden, 1, 1, act=act), nn.Conv2d(hidden, 1, 1, bias=True), nn.Sigmoid())
        self.channel_gate = OmniLiteChannelGate(c2, reduction=reduction)
        self.spatial_gate = OmniLiteSpatialGate(k=3)
        self.local_branch = Conv(c2, c2, 3, 1, g=c2, act=act)
        self.context_branch = Conv(c2, c2, 5, 1, g=c2, act=act)
        self.low_pool = nn.AvgPool2d(3, 1, 1)
        self.high_refine = Conv(c2, c2, 3, 1, g=c2, act=act)
        self.selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, 3, 1, bias=True), nn.Softmax(dim=1))
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def _ghost_feature(self, x):
        y = self.primary(x)
        outs = [y]
        if not isinstance(self.cheap3, nn.Identity):
            outs.append(self.cheap3(y))
        if not isinstance(self.cheap5, nn.Identity):
            outs.append(self.cheap5(y))
        y = torch.cat(outs, dim=1)
        return y[:, :self.c2, :, :]

    def forward(self, x):
        y = self._ghost_feature(x)
        scale, bias = self.remap(y).chunk(2, dim=1)
        y = y * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(bias)
        sparse = self.sparse_score(y)
        c_gate = self.channel_gate(y)
        s_gate = self.spatial_gate(y)
        gate = sparse * c_gate * s_gate
        local = self.local_branch(y)
        context = self.context_branch(y)
        low = self.low_pool(y)
        high = self.high_refine(y - low)
        w = self.selector(y)
        enhance = w[:, 0:1] * local + w[:, 1:2] * context + w[:, 2:3] * high
        return y + self.gamma * gate * enhance

class OmniLiteGhostBottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = OmniLiteGhostConv(c1, c2, 1, 1)
        self.cv2 = OmniLiteGhostConv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3OmniLiteGhost(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(OmniLiteGhostBottleneck(c_, c_, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class OmniLiteIllumGhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, reduction=8, variant='full'):
        super().__init__()
        self.variant = str(variant)
        c_primary = max(c2 // 2, 1)
        c_ghost = c2 - c_primary
        self.primary = Conv(c1, c_primary, k, s, g=g, act=act)
        c_g3 = c_ghost // 2
        c_g5 = c_ghost - c_g3
        self.cheap3 = Conv(c_primary, c_g3, 3, 1, g=_gcd_group(c_primary, c_g3), act=act) if c_g3 > 0 else nn.Identity()
        self.cheap5 = Conv(c_primary, c_g5, 5, 1, g=_gcd_group(c_primary, c_g5), act=act) if c_g5 > 0 else nn.Identity()
        self.c2 = c2
        hidden = max(c2 // reduction, 8)
        self.remap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, hidden, 1, bias=False), nn.SiLU(inplace=True), nn.Conv2d(hidden, c2 * 2, 1, bias=True))
        self.sparse_score = nn.Sequential(Conv(c2, hidden, 1, 1, act=act), nn.Conv2d(hidden, 1, 1, bias=True), nn.Sigmoid())
        self.channel_gate = OmniLiteChannelGate(c2, reduction=reduction)
        self.spatial_gate = OmniLiteSpatialGate(k=3)
        self.local_branch = Conv(c2, c2, 3, 1, g=c2, act=act)
        self.context_branch = Conv(c2, c2, 5, 1, g=c2, act=act)
        self.low_pool = nn.AvgPool2d(3, 1, 1)
        self.high_refine = Conv(c2, c2, 3, 1, g=c2, act=act)
        use_illum = self.variant not in {'local_only', 'context_only', 'high_only', 'no_illum', 'no_enhance'}
        self.illum_gate = nn.Sequential(nn.Conv2d(2, 8, 1, bias=False), nn.SiLU(inplace=True), nn.Conv2d(8, c2, 1, bias=True), nn.Sigmoid()) if use_illum else None
        selector_outputs = 3 if self.variant == 'no_illum' else 4
        self.selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, selector_outputs, 1, bias=True), nn.Softmax(dim=1)) if self.variant not in {'local_only', 'context_only', 'high_only', 'illum_only', 'equal_mix', 'no_enhance'} else None
        self.gamma = nn.Parameter(torch.ones(1) * 0.1) if self.variant != 'no_enhance' else None

    def _ghost_feature(self, x):
        y = self.primary(x)
        outs = [y]
        if not isinstance(self.cheap3, nn.Identity):
            outs.append(self.cheap3(y))
        if not isinstance(self.cheap5, nn.Identity):
            outs.append(self.cheap5(y))
        y = torch.cat(outs, dim=1)
        return y[:, :self.c2, :, :]

    def forward(self, x):
        y = self._ghost_feature(x)
        if self.variant != 'no_remap':
            scale, bias = self.remap(y).chunk(2, dim=1)
            y = y * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(bias)
        if self.variant == 'no_enhance':
            return y
        sparse = 1.0 if self.variant == 'no_sparse' else self.sparse_score(y)
        c_gate = 1.0 if self.variant == 'no_channel' else self.channel_gate(y)
        s_gate = 1.0 if self.variant == 'no_spatial' else self.spatial_gate(y)
        gate = sparse * c_gate * s_gate
        local = self.local_branch(y) if self.local_branch is not None else None
        context = self.context_branch(y) if self.context_branch is not None else None
        high = self.high_refine(y - self.low_pool(y)) if self.high_refine is not None else None
        if self.illum_gate is not None:
            mean = x.mean(dim=1, keepdim=True)
            std = torch.sqrt((x - mean).pow(2).mean(dim=1, keepdim=True) + 1e-06)
            cue = torch.cat([mean, std], dim=1)
            cue = F.interpolate(cue, size=y.shape[-2:], mode='bilinear', align_corners=False)
            illum = self.illum_gate(cue) * y
        else:
            illum = None
        if self.variant == 'local_only':
            enhance = local
        elif self.variant == 'context_only':
            enhance = context
        elif self.variant == 'high_only':
            enhance = high
        elif self.variant == 'illum_only':
            enhance = illum
        elif self.variant == 'equal_mix':
            enhance = (local + context + high + illum) * 0.25
        else:
            w = self.selector(y)
            if self.variant == 'no_illum':
                enhance = w[:, 0:1] * local + w[:, 1:2] * context + w[:, 2:3] * high
            else:
                enhance = w[:, 0:1] * local + w[:, 1:2] * context + w[:, 2:3] * high + w[:, 3:4] * illum
        return y + self.gamma * gate * enhance

class OmniLiteIllumGhostBottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, reduction=8, variant='full'):
        super().__init__()
        self.cv1 = OmniLiteIllumGhostConv(c1, c2, 1, 1, reduction=reduction, variant=variant)
        self.cv2 = OmniLiteIllumGhostConv(c2, c2, 3, 1, reduction=reduction, variant=variant)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3OmniLiteIllumGhost(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, reduction=8, variant='full'):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(OmniLiteIllumGhostBottleneck(c_, c_, shortcut, reduction, variant) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class LCGF(nn.Module):

    def __init__(self, c1_rgb, c1_t, c2, groups=4, reduction=4, use_spatial=True, variant='full'):
        super().__init__()
        self.variant = str(variant)
        c_mid = c2
        groups = max(int(groups), 1)
        if c_mid % groups != 0:
            groups = math.gcd(c_mid, groups) or 1
        self.c_mid = c_mid
        self.c2 = c2
        self.groups = groups
        self.group_channels = c_mid // groups
        self.use_spatial = bool(use_spatial)
        self.rgb_align = nn.Identity() if c1_rgb == c_mid else Conv(c1_rgb, c_mid, 1, 1)
        self.t_align = nn.Identity() if c1_t == c_mid else Conv(c1_t, c_mid, 1, 1)
        hidden = max(c_mid // max(int(reduction), 1), 8)
        self.quality_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_mid * 3, hidden, 1, bias=False), nn.SiLU(inplace=True), nn.Conv2d(hidden, 2, 1, bias=True)) if self.variant != 'equal_quality' else None
        self.diff_enhance = Conv(c_mid, c_mid, 3, 1, g=c_mid) if self.variant not in {'no_diff_enhance', 'base_only'} else None
        gate_in = groups * 2
        gate_hidden = max(groups, gate_in // max(int(reduction), 1), 4)
        self.group_gate = nn.Sequential(nn.Linear(gate_in, gate_hidden, bias=False), nn.SiLU(inplace=True), nn.Linear(gate_hidden, groups, bias=True), nn.Sigmoid()) if self.variant not in {'no_group', 'base_only'} else None
        self.spatial_gate = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False), nn.Sigmoid()) if self.use_spatial and self.variant not in {'no_spatial', 'base_only'} else None
        self.beta = nn.Parameter(torch.ones(1) * 0.1) if self.variant not in {'no_recal', 'discrepancy_only', 'base_only'} else None
        self.alpha = nn.Parameter(torch.ones(1) * 0.1) if self.variant not in {'no_recal', 'consensus_only', 'base_only'} else None
        self.gamma = nn.Parameter(torch.ones(1) * 0.1) if self.variant != 'base_only' else None
        self.out = GhostConv(c_mid, c2, 1, 1)

    def _spatial_response(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(avg + mx)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError('LCGF expects input as [rgb_feature, thermal_feature].')
        r, t = (x[0], x[1])
        r = self.rgb_align(r)
        t = self.t_align(t)
        diff_raw = r - t
        diff_abs = diff_raw.abs()
        if self.variant == 'equal_quality':
            wr = wt = 0.5
        else:
            q = self.quality_gate(torch.cat((r, t, diff_abs), dim=1))
            q = torch.softmax(q, dim=1)
            wr = q[:, 0:1]
            wt = q[:, 1:2]
        base = wr * r + wt * t
        if self.variant == 'base_only':
            return self.out(base)
        a_rgb = self._spatial_response(r)
        a_ir = self._spatial_response(t)
        m_cons = a_rgb * a_ir
        m_dis = a_ir * (1.0 - a_rgb)
        if self.variant == 'no_recal':
            recal = torch.ones_like(m_cons)
        elif self.variant == 'consensus_only':
            recal = 1.0 + self.beta * m_cons
        elif self.variant == 'discrepancy_only':
            recal = 1.0 + self.alpha * m_dis
        else:
            recal = (1.0 + self.beta * m_cons) * (1.0 + self.alpha * m_dis)
        diff_enhanced = diff_raw if self.variant == 'no_diff_enhance' else self.diff_enhance(diff_raw)
        b, c, h, w = diff_abs.shape
        g = self.groups
        cg = self.group_channels
        diff_group = diff_abs.reshape(b, g, cg, h, w)
        diff_avg = diff_group.mean(dim=(2, 3, 4))
        diff_max = diff_group.amax(dim=(2, 3, 4))
        if self.variant == 'no_group':
            group_gate = torch.ones((b, g, 1, 1, 1), device=diff_abs.device, dtype=diff_abs.dtype)
        else:
            group_gate = self.group_gate(torch.cat((diff_avg, diff_max), dim=1)).reshape(b, g, 1, 1, 1)
        diff_enhanced = diff_enhanced.reshape(b, g, cg, h, w)
        base = base.reshape(b, g, cg, h, w)
        recal = recal.reshape(b, 1, 1, h, w)
        if self.spatial_gate is not None and self.variant != 'no_spatial':
            spatial_avg = diff_abs.mean(dim=1, keepdim=True)
            spatial_max = diff_abs.amax(dim=1, keepdim=True)
            spatial_gate = self.spatial_gate(torch.cat((spatial_avg, spatial_max), dim=1)).reshape(b, 1, 1, h, w)
            gate = group_gate * spatial_gate * recal
        else:
            gate = group_gate * recal
        fused = base + self.gamma * gate * diff_enhanced
        fused = fused.reshape(b, c, h, w)
        return self.out(fused)

class ECAGate(nn.Module):

    def __init__(self, k_size=3):
        super().__init__()
        k_size = int(k_size)
        if k_size % 2 == 0:
            k_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv1d(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return self.act(y)

class SpatialGate(nn.Module):

    def __init__(self, k_size=3):
        super().__init__()
        p = k_size // 2
        self.net = nn.Sequential(nn.Conv2d(2, 1, kernel_size=k_size, stride=1, padding=p, bias=False), nn.Sigmoid())

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        return self.net(torch.cat([avg_map, max_map], dim=1))

class DGEC3Ghost(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, eca_k=3, spatial_k=3, gamma_init=0.0, variant='full'):
        super().__init__()
        self.variant = str(variant)
        self.c3ghost = C3Ghost(c1, c2, n, shortcut)
        self.channel_gate = ECAGate(k_size=eca_k) if self.variant in {'full', 'channel_only', 'no_detail'} else None
        self.spatial_gate = SpatialGate(k_size=spatial_k) if self.variant in {'full', 'spatial_only', 'no_detail'} else None
        self.detail = Conv(c2, c2, k=3, s=1, g=c2) if self.variant not in {'c3ghost_only', 'no_detail'} else None
        self.gamma = nn.Parameter(torch.ones(1) * gamma_init) if self.variant != 'c3ghost_only' else None

    def forward(self, x):
        y = self.c3ghost(x)
        if self.variant == 'c3ghost_only':
            return y
        gc = self.channel_gate(y) if self.channel_gate is not None else None
        gs = self.spatial_gate(y) if self.spatial_gate is not None else None
        e = y if self.variant == 'no_detail' else self.detail(y)
        if self.variant == 'detail_only':
            enhance = e
        elif self.variant == 'channel_only':
            enhance = gc * e
        elif self.variant == 'spatial_only':
            enhance = gs * e
        else:
            enhance = gc * gs * e
        out = y + self.gamma * enhance
        return out
