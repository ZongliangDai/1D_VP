import numpy as np
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy.fft as fft


# ----------------------------
# 1. 全局物理常量（与原代码一致）
# ----------------------------
MASS_ELECTRON   = 9.1094e-31      # kg
MASS_PROTON     = 1.6726485e-27   # kg
CHARGE_ELECTRON = 1.6021892e-19   # C
DENSITY         = 1.0e19          # m^-3
TEMPERATURE     = 1.0             # keV
EPSILON         = 8.854187817e-12 # F/m
MAGNETIC        = 1.0             # Tesla

M_REF = MASS_ELECTRON
E_REF = CHARGE_ELECTRON
N_REF = DENSITY
T_REF = TEMPERATURE * 1000.0 * CHARGE_ELECTRON
V_REF = np.sqrt(T_REF / M_REF)
TIME_REF = np.sqrt(EPSILON * M_REF / (N_REF * E_REF**2))
X_REF = V_REF * TIME_REF
EPSILON_REF = 1.0


# ----------------------------
# 2. 模拟参数配置（与原代码一致）
# ----------------------------
class SimulationParams:
    def __init__(self):
        self.flag_nonlinear = True
        self.e = -1.0
        self.m = 1.0
        self.k0 = 0.4
        self.delta = 0.05
        self.k_num = 32
        self.v_num = 513
        self.v_max = 6.0
        self.dt = 0.1
        self.total_time = 200.0
        self.diag_point_num = 500
        self.e_m = self.e / self.m
        self.x_num = 4 * self.k_num
        self.diag_time = self.total_time / self.diag_point_num


# ----------------------------
# 3. 傅里叶变换工具（修正对称填充错误）
# ----------------------------
class FourierTransformer:
    @staticmethod
    def f2fk(f, k_num):
        # 与原代码一致：实空间->波数空间
        fk_ex = fft.ifft(f, axis=0, workers=-1)
        fk = np.conj(fk_ex[:k_num, :])
        return fk

    @staticmethod
    def fk2f(fk, x_num):
        # 修正对称填充逻辑，与原代码完全一致
        k_num, v_num = fk.shape
        fk_ex = np.zeros((x_num, v_num), dtype=np.complex128)  # 用complex128保证精度
        fk_ex[:k_num, :] = fk * x_num
        # 原代码逻辑：-1:-k_num:-1 是从最后一个元素倒序取到k_num个（不含-k_num）
        fk_ex[-1:-k_num:-1, :] = np.conj(fk[1:k_num, :] * x_num)  # 修正索引顺序
        f = np.real(fft.ifft(fk_ex, axis=0, workers=-1))
        return f


# ----------------------------
# 4. 核心算法（还原原代码逻辑）
# ----------------------------
class SimulationCore:
    def __init__(self, params, transformer):
        self.params = params
        self.ft = transformer

    def upd(self, pre_or_cor, upd_phase, dfk0):
        return dfk0 * upd_phase[:, :, pre_or_cor]

    def pull_back(self, flag_nl, pre_or_cor, phik_t, gkx_phik, gkv_phik, dv_f0, dfk, k, hv, x_num, k_num):
        # 完全复刻原代码的物理逻辑，仅保留必要的向量化
        n = 1 if pre_or_cor else 0
        gkv = gkv_phik[:, :, n] * phik_t

        if not flag_nl:
            dfk += gkv * dv_f0
            return dfk

        gkx = gkx_phik[:, :, n] * phik_t
        gx = self.ft.fk2f(gkx, x_num)
        gv = self.ft.fk2f(gkv, x_num)

        # 非线性项1：严格按原代码步骤计算
        dfk_dx = 1j * k * dfk
        dfk_dv = np.zeros_like(dfk, dtype=np.complex128)
        dfk_dv[:, 0] = (dfk[:, 1] - dfk[:, 0]) * hv
        dfk_dv[:, 1:-1] = (dfk[:, 2:] - dfk[:, :-2]) * 0.5 * hv
        dfk_dv[:, -1] = (dfk[:, -1] - dfk[:, -2]) * hv
        dfk_dv[0, :] += dv_f0[0, :]

        df_dx = self.ft.fk2f(dfk_dx, x_num)
        df_dv = self.ft.fk2f(dfk_dv, x_num)
        dfa = gx * df_dx + gv * df_dv
        dfk_a = self.ft.f2fk(dfa, k_num)

        # 非线性项2：严格按原代码步骤计算
        dfk_dx = 1j * k * dfk_a
        dfk_dv[:, 0] = (dfk_a[:, 1] - dfk_a[:, 0]) * hv
        dfk_dv[:, 1:-1] = (dfk_a[:, 2:] - dfk_a[:, :-2]) * 0.5 * hv
        dfk_dv[:, -1] = (dfk_a[:,-1] - dfk_a[:, -2]) * hv

        df_dx = self.ft.fk2f(dfk_dx, x_num)
        df_dv = self.ft.fk2f(dfk_dv, x_num)
        dfb = 0.5 * (gx * df_dx + gv * df_dv)
        dfk_b = self.ft.f2fk(dfb, k_num)

        dfk += dfk_a + dfk_b
        return dfk

    def field(self, dfk, e, k, dv):
        # 与原代码一致的电场计算
        dnk = np.sum(dfk, axis=1).reshape(-1, 1) * dv
        return np.where(k < 0.1 * k[1, 0], 0.0, e * dnk / (EPSILON_REF * k**2))


# ----------------------------
# 5. 诊断输出（确保数据对应正确时间步）
# ----------------------------
class Diagnostics:
    def __init__(self, params):
        self.params = params
        self.output_dir = 'Output'
        os.makedirs(self.output_dir, exist_ok=True)
        self.ft = None
        self._files = {}  # 用字典管理文件句柄

    def set_transformer(self, transformer):
        self.ft = transformer

    def _open_files(self):
        # 打开文件句柄（确保写入正确）
        self._files['time'] = open(f'{self.output_dir}/TimeList.dat', 'w')
        self._files['ene'] = open(f'{self.output_dir}/energy.dat', 'w')
        self._files['phi'] = open(f'{self.output_dir}/phi.dat', 'wb')
        self._files['phik'] = open(f'{self.output_dir}/phik.dat', 'wb')
        self._files['fa'] = open(f'{self.output_dir}/fa.dat', 'wb')

    def init_files(self, k, x, v, f0):
        # 与原代码一致的初始化文件
        np.savetxt(f'{self.output_dir}/Normalization.dat', [
            M_REF, E_REF, TIME_REF, V_REF, X_REF,
            self.params.k0, self.params.total_time, self.params.dt
        ])
        k.tofile(f'{self.output_dir}/k.bin')
        x.tofile(f'{self.output_dir}/x.bin')
        v.tofile(f'{self.output_dir}/v.bin')
        f0.tofile(f'{self.output_dir}/F0.bin')
        self._open_files()  # 初始化时打开文件

    def record(self, t, phik, dfk, k0, x_num):
        # 确保诊断数据与时间步严格对应
        self._files['time'].write(f'{t:.6f}\n')
        ene = np.sum(np.abs(phik)** 2) * np.pi / k0
        self._files['ene'].write(f'{ene:.6e}\n')
        phi = self.ft.fk2f(phik, x_num)
        phi.tofile(self._files['phi'])
        np.abs(phik).tofile(self._files['phik'])
        np.abs(dfk[0, :]).tofile(self._files['fa'])

    def close(self):
        for f in self._files.values():
            f.close()


# ----------------------------
# 6. 初始化模拟（还原原代码网格与初始条件）
# ----------------------------
def initialize_simulation(params):
    # 波数网格（向量化生成，确保内存连续）
    k = np.linspace(0, (params.k_num-1) * params.k0, params.k_num, dtype=np.float64).reshape(-1, 1)
    
    # 速度网格
    v = np.linspace(-params.v_max, params.v_max, params.v_num, dtype=np.float64).reshape(1, -1)
    
    # 预计算常用系数（避免后续重复计算）
    dv = v[0, 1] - v[0, 0]  # 直接从linspace获取，精度更高
    hv = 1.0 / dv
    
    # 平衡分布函数（向量化计算，减少中间变量）
    f0 = np.sqrt(1.0 / (2.0 * np.pi)) * np.exp(-0.5 * v*v)
    
    # F0的导数（向量化差分，使用切片操作替代循环）
    dv_f0 = np.zeros_like(f0, dtype=np.complex128)
    dv_f0[0, 1:-1] = (f0[0, 2:] - f0[0, :-2]) * 0.5 * hv  # 中间点
    dv_f0[0, 0] = (f0[0, 1] - f0[0, 0]) * hv  # 左边界
    dv_f0[0, -1] = (f0[0, -1] - f0[0, -2]) * hv  # 右边界
    
    # 初始扰动（向量化赋值）
    dfk = np.zeros((params.k_num, params.v_num), dtype=np.complex128)
    dfk[1, :] = params.delta * 0.5 * f0
    
    # 预计算时间推进相位因子（向量化广播）
    k_ex = k[:, :, np.newaxis]
    v_ex = v[:, :, np.newaxis]
    dtc = np.array([0.5 * params.dt, params.dt]).reshape(1, 1, 2)
    upd_phase = np.exp(-1j * k_ex * v_ex * dtc).astype(np.complex128)
    
    # 预计算Gkx和Gkv（向量化条件判断）
    kv = k_ex * v_ex
    condition = np.abs(kv) < 0.1 * dv
    gkv_phik = np.where(
        condition,
        params.e_m * 1j * k_ex * dtc,
        params.e_m * (1.0 - np.exp(-1j * kv * dtc)) / v_ex
    ).astype(np.complex128)
    gkx_phik = np.where(
        condition,
        0.5 * gkv_phik * dtc,
        gkv_phik / (1j * kv) - params.e_m * dtc * np.exp(-1j * kv * dtc) / v_ex
    ).astype(np.complex128)
    
    # 初始电场
    core = SimulationCore(params, FourierTransformer())
    phik = core.field(dfk, params.e, k, dv)
    
    # 实空间网格（向量化生成）
    dx = 2.0 * np.pi / (params.k0 * params.x_num)
    x = np.linspace(0, (params.x_num-1) * dx, params.x_num, dtype=np.float64).reshape(-1, 1)
    
    # 诊断时间点（向量化判断）
    time_steps = np.arange(0, params.total_time, params.dt)
    is_diag_time = np.isclose(
        np.mod(time_steps + 0.1 * params.dt, params.diag_time),
        0, atol=0.2 * params.dt
    )
    
    # 确保所有数组内存连续（提升缓存利用率）
    return {
        'k': np.ascontiguousarray(k),
        'x': np.ascontiguousarray(x),
        'v': np.ascontiguousarray(v),
        'dv': dv, 'hv': hv,
        'f0': np.ascontiguousarray(f0),
        'dv_f0': np.ascontiguousarray(dv_f0),
        'dfk': np.ascontiguousarray(dfk),
        'phik': np.ascontiguousarray(phik),
        'upd_phase': np.ascontiguousarray(upd_phase),
        'gkx_phik': np.ascontiguousarray(gkx_phik),
        'gkv_phik': np.ascontiguousarray(gkv_phik),
        'time_steps': time_steps,
        'is_diag_time': is_diag_time
    }


# ----------------------------
# 7. 主流程（严格复刻原代码时间步进）
# ----------------------------
def run_simulation():
    params = SimulationParams()
    ft = FourierTransformer()
    core = SimulationCore(params, ft)
    diagnostics = Diagnostics(params)
    diagnostics.set_transformer(ft)

    sim_data = initialize_simulation(params)
    dfk = sim_data['dfk'].copy()  # 初始复制必要
    phik = sim_data['phik'].copy()
    k = sim_data['k']
    hv = sim_data['hv']
    dv = sim_data['dv']
    dv_f0 = sim_data['dv_f0']
    upd_phase = sim_data['upd_phase']
    gkx_phik = sim_data['gkx_phik']
    gkv_phik = sim_data['gkv_phik']
    time_steps = sim_data['time_steps']
    is_diag_time = sim_data['is_diag_time']

    diagnostics.init_files(sim_data['k'], sim_data['x'], sim_data['v'], sim_data['f0'])
    diagnostics.record(0.0, phik, dfk, params.k0, params.x_num)

    # 预分配dfk0缓冲区，避免重复内存申请（核心优化）
    dfk0 = np.empty_like(dfk)
    np.copyto(dfk0, dfk)  # 初始填充

    start_time = time.perf_counter()
    progress_bar = tqdm(
        total=params.total_time,
        unit=f"$\\omega_\\text{{pe}}^{{-1}}$",
        desc='模拟进度',
        bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    # 提取参数减少属性访问
    flag_nl = params.flag_nonlinear
    x_num, k_num = params.x_num, params.k_num

    for step, t in enumerate(time_steps):
        progress_bar.update(params.dt)

        # 预测步：仅在非线性模式下需要复制dfk到dfk0
        if flag_nl:
            pre_or_cor = 0
            # 用copyto替代dfk0 = dfk.copy()，减少内存碎片
            np.copyto(dfk0, dfk)
            # 时间推进（直接在dfk上操作，避免额外复制）
            dfk[:] = core.upd(pre_or_cor, upd_phase, dfk0)  # 原地修改
            # 物理计算
            dfk = core.pull_back(
                flag_nl, pre_or_cor, phik, gkx_phik, gkv_phik,
                dv_f0, dfk, k, hv, x_num, k_num
            )
            # 更新电场
            phik = core.field(dfk, params.e, k, dv)
        
        # 校正步：复用预测步的dfk0（非线性模式），避免重复复制
        pre_or_cor = 1
        if not flag_nl:
            # 线性模式下才需要复制，非线性模式直接用预测步的dfk0
            np.copyto(dfk0, dfk)
        # 时间推进（原地修改）
        dfk[:] = core.upd(pre_or_cor, upd_phase, dfk0)
        # 物理计算
        dfk = core.pull_back(
            flag_nl, pre_or_cor, phik, gkx_phik, gkv_phik,
            dv_f0, dfk, k, hv, x_num, k_num
        )
        # 更新电场
        phik = core.field(dfk, params.e, k, dv)

        # 诊断输出
        if is_diag_time[step]:
            diagnostics.record(t + params.dt, phik, dfk, params.k0, x_num)

    diagnostics.close()
    progress_bar.close()
    total_elapsed = time.perf_counter() - start_time
    print(f"模拟完成 | 总耗时 {total_elapsed:.2f} 秒")
    plot_results(diagnostics.output_dir)


# ----------------------------
# 8. 结果可视化（与原代码一致）
# ----------------------------
def plot_results(output_dir):
    t = np.loadtxt(f'{output_dir}/TimeList.dat')
    ene = np.loadtxt(f'{output_dir}/energy.dat')

    fig, ax = plt.subplots()
    ax.plot(t, np.log10(ene), color='green', linewidth=1.5, label='$\\lg |E|^2$')
    ax.set_xlabel('$\\omega_{\\text{pe}} t$', fontsize=12)
    ax.set_ylabel('$\\lg |E|^2$', fontsize=12)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-4, -1)
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    plt.savefig('energy_evolution.jpg', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"  # 禁用多线程避免潜在的数值差异
    run_simulation()