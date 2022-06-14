---
title: 使用Docker搭建GitLab
description: 使用Docker搭建GitLab
categories:
 - Git
tags:
 - gitlab
---

### 运行容器

```bash
export GITLAB_HOME=/srv/gitlab
docker run -it -d \
  -p 23080:80 \
  -p 23022:22 \
  -v $GITLAB_HOME/config:/etc/gitlab \
  -v $GITLAB_HOME/logs:/var/log/gitlab \
  -v $GITLAB_HOME/data:/var/opt/gitlab \
  -m 2048m \
  --memory-swap=2048m \
  --restart always \
  --name gitlab \
  gitlab/gitlab-ce:latest
```

### 修改配置

```
# vi $GITLAB_HOME/config/gitlab.rb
external_url 'http://${宿主机IP}'  # http IP
gitlab_rails['gitlab_ssh_host'] = '${宿主机IP}'  # ssh IP
gitlab_rails['gitlab_shell_ssh_port'] = 23022  # ssh 端口
puma['worker_timeout'] = 60  # 应用服务连接超时
puma['worker_processes'] = 2  # 应用服务连接数
postgresql['max_worker_processes'] = 8   # 数据库连接数
postgresql['shared_buffers'] = "256MB"  # 数据库内存
prometheus_monitoring['enable'] = false  # 关闭监控
```

### 配置生效

```bash
docker exec -it gitlab gitlab-ctl reconfigure
```

***速度较慢***

### 修改端口

```bash
docker exec -it gitlab vi /opt/gitlab/embedded/service/gitlab-rails/config/gitlab.yml
```

```yaml
gitlab:
  host: ${宿主机IP}
  port: ${宿主机端口}
  https: false
```

### 重启服务

```bash
docker exec -it gitlab gitlab-ctl restart
```

### 修改root密码

```bash
docker exec -it gitlab bash
gitlab-rails console -e production
user=User.where(id:1).first
user.password='passwd'
user.save!
exit
```

