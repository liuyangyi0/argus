# Linux 容器部署与验证

本入口面向 Linux x86_64。镜像包含构建后的 Vue 工作台、已锁定的 Python 依赖、
FFmpeg 和固定版本的 [go2rtc 1.9.14](https://github.com/AlexxIT/go2rtc/releases/tag/v1.9.14)。
这份配置不代表真实摄像头、GPU 检测或 production 门禁已经验收通过。
Jetson/ARM 仍需独立的设备依赖与验证；不能套用 x86_64 锁文件。

## 首次配置

1. 在仓库外创建私有配置目录，将 `configs/default.yaml` 复制为 `runtime.yaml`。
   按设备填写摄像头 `source` 与 `protocol`、场景目录和模型路径；不要把凭证提交进仓库。
2. 容器以 UID/GID `10001:10001` 运行。私有配置目录及其中 YAML 必须允许该用户读写，
   因为场景选择和配置管理需要原子写入。挂载整个目录，不能只挂载单个 YAML 文件。
3. 设置进程环境变量 `ARGUS_CONFIG_DIR` 为该目录绝对路径，`ARGUS_API_TOKEN` 为强随机值
   （建议至少 32 字节随机数据）。Compose 缺少任意一个值时会拒绝展开配置。
   Docker Compose 自身的环境文件机制与 Argus 的配置加载是两回事。
4. 保持运行数据在 `/app/data` 下；Compose 使用命名卷持久化整个目录，包含数据库、
   模型、场景 profile、录像、复核材料及备份。已有宿主机 `./data` 不会自动迁入新卷，
   首次迁移须先做停机备份，复制后核对摘要及 UID/GID，禁止把空卷当成旧数据已恢复。
5. NVIDIA 部署显式配置 `torch-cuda` 和授权 `.pt` 工件，另行配置 NVIDIA 容器设备映射。
   USB 部署另行配置设备映射和 video 组权限；默认 Compose 不授予设备或 privileged 权限。
   模型加载失败不允许自动切换为 CPU/OpenVINO/SSIM。

先检查配置，再启动（完整展开的配置含 token，不应公开日志）：

```sh
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
```

默认仅将主机 `127.0.0.1:8080` 映射到容器。容器内部监听 `0.0.0.0:8080` 并强制开启认证。
通过本机浏览器或 SSH 隧道完成初始化，立即更换首次创建的 `admin/admin` 密码，
再配置独立管理员/工程账号；人工审批必须使用个人账号登录，不能用共享 API token。
未配置 `ARGUS_SESSION_SECRET` 时，每次服务启动使用新的随机会话密钥，重启后必须重新登录。
如需跨重启保持登录，在私有 Compose override 的容器环境中设置独立的
`ARGUS_SESSION_SECRET`（至少 32 字节随机数据）；不得复用共享 API token，也不要将其提交进仓库。
需要远程访问时，在反向代理层配置 TLS 和访问控制，不直接发布 go2rtc 管理端口。
WebRTC 的媒体端口/候选地址需要按部署网络单独配置；未配置时保留同源 MSE/MJPEG 降级。

## 健康、数据与关闭

Docker healthcheck 仅确认 HTTP 服务响应，不能证明场景、模型或检测可用。
检测状态应查看 `/api/system/health/ready` 及工作台的 SceneGuard/模型状态。
显式 `dashboard.frontend_dist_dir` 缺失 `index.html` 时启动报错，避免只有 API 的镜像假装完整交付。

`docker compose down` 保留命名数据卷；不要使用 `down --volumes` 清理运行数据。
配置目录是可写私有绑定目录，需与数据卷一起备份。恢复后先核对模型/profile/artifact 身份，
再检查场景恢复状态，不手工改数据库 stage 绕过发布服务。

## 无训练、无识别的镜像检查

```sh
docker build --target functional-check -t argus-functional-check .
docker run --rm --network none argus-functional-check
```

该目标只安装普通功能依赖，先从已安装 wheel 请求构建后的 SPA 和 JavaScript 资源，
再运行受控 API/配置测试；模型引擎及摄像头、源视频访问被禁止。
CI 自动执行此检查。正式 runtime 镜像及 CUDA 链路仍要在目标设备验证，之后保存镜像 digest，
完成独立视频回归和可信长稳验收才可申请 production。
