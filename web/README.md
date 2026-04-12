# Argus Web Dashboard

这是 Argus 的前端子项目，使用 Vue 3、TypeScript、Vite、Pinia 和 Ant Design Vue。

## 当前页面结构

- `/overview`：系统总览。
- `/cameras`：摄像头列表。
- `/cameras/:id`：摄像头详情。
- `/alerts`：告警列表与处理。
- `/models`：基线、训练、模型、对比、标注、阈值预览。
- `/system`：系统概览、配置管理、用户、审计、降级历史、备份、音频配置、存储清理。

## 本地开发

```powershell
cd web
npm install
npm run dev
```

默认由 Vite 提供开发服务；生产构建产物会由后端 FastAPI 通过 `web/dist` 托管。

## 构建

```powershell
cd web
npm run build
```

## 主要依赖

- Vue 3
- Vue Router
- Pinia
- Ant Design Vue
- Axios
- ECharts / vue-echarts

## 与后端的关系

- 前端只消费 `src/argus/dashboard/app.py` 中已注册的 `/api/*` 接口和 `/ws` WebSocket。
- 系统管理类能力很多是“系统页子标签 + 后端 API”的组合，不是独立前端路由。
- 新增页面前，先确认后端 API 已存在且数据模型稳定。
