import m5
from m5.objects import *
from m5.objects.MatrixFlowEngine import MatrixFlowEngine

# 1. 创建最基础的系统与时钟 (1GHz)
system = System()
system.clk_domain = SrcClockDomain(clock='1GHz', voltage_domain=VoltageDomain())
system.mem_mode = 'timing'

# 2. 创建假内存 + 总线（供 DMA 与 system_port 共享）
system.membus = IOXBar()
system.mem_ctrl = SimpleMemory(range=AddrRange('512MB'))
system.mem_ctrl.port = system.membus.mem_side_ports

# 3. 实例化 MatrixFlow 引擎
system.matrix_engine = MatrixFlowEngine(
    mac_array_size=16,
    compute_latency_per_op=1
)
system.matrix_engine.dma_port = system.membus.cpu_side_ports

# 4. system_port 连到总线
system.system_port = system.membus.cpu_side_ports

# 5. 启动仿真
root = Root(full_system=False, system=system)
m5.instantiate()

print("--- 仿真开始 ---")
# 1024^3/(16*16)*1 周期 ≈ 4194304 周期，跑 50 亿 Ticks 以覆盖 processComputeDone
exit_event = m5.simulate(5000000000)
print(f"--- 仿真结束 @ tick {m5.curTick()} 因为 {exit_event.getCause()} ---")
