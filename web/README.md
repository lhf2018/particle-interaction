# FluidVibe Web

前端页面由 FastAPI 提供服务，定位为轻量预览（非核心 3D 渲染入口）。
当前架构：前端发送输入到 `/ws/sim` 或 `/ws/sim3d`，后端通过 WebSocket 实时推送粒子坐标。

## 启动

在项目根目录执行：

```bash
uvicorn app:app --reload --port 8000
```

然后打开：

`http://localhost:8000/web/`

默认是 2D 预览模式，3D 预览：

`http://localhost:8000/web/index.html?mode=3d`

## 本地 3D 推荐入口（核心体验）

```bash
python fluid_sim_3d.py --particles 120000 --width 1280 --height 720
```

健康检查接口：

`http://localhost:8000/api/health`

## 交互

- 左键按住：排斥
- 右键按住：吸引
- 滚轮：调节力强度
- `Space`：暂停/继续
- `R`：重置粒子
