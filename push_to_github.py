import os
import shutil
import subprocess

# ==================== 配置区 ====================
GITHUB_URL = "https://github.com/749894867/LLM.git"

GITIGNORE_CONTENT = """
.venv/
venv/
__pycache__/
*.pth
*.pt
*.bin
*.ckpt*
models/
checkpoint/
.idea/
.vscode/
*.log
"""


# ===============================================

def run_cmd(command):
    """执行系统命令并打印结果"""
    print(f"正在执行: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout.strip())
        return True
    else:
        print(f"❌ 出错啦: {result.stderr.strip()}")
        return False


def auto_push():
    print("🚀 开始全新一键配置并上传到 GitHub...")

    # 1. 自动写入严格的 .gitignore
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(GITIGNORE_CONTENT.strip())
    print("✅ 1. 新的 .gitignore 过滤文件创建成功！")

    # 2. 初始化本地 Git 仓库
    if not run_cmd("git init"): return
    print("✅ 2. 本地 Git 初始化成功！")

    # 3. 精准添加文件
    if not run_cmd("git add *.py .gitignore"): return
    print("✅ 3. 代码文件已成功添加到暂存区（大文件已完美绕过）！")

    # 4. 提交到本地
    if not run_cmd('git commit -m "first clean commit by auto script"'):
        # 如果提示没有文件要提交，也算成功，继续往下走
        pass
    print("✅ 4. 本地打包提交完成！")

    # 5. 创建并切换到 main 分支
    if not run_cmd("git branch -M main"): return
    print("✅ 5. 已自动创建并切换至 main 分支！")

    # 6. 【核心修复】强行清理旧远程关联，防止 already exists 报错
    subprocess.run("git remote remove origin", shell=True, capture_output=True)
    if not run_cmd(f"git remote add origin {GITHUB_URL}"): return
    print("✅ 6. 已成功关联 GitHub 远程仓库！")

    # 7. 推送到 GitHub
    print("\n------------------------------------------------------------")
    print("🛫 正在向 GitHub 服务器推送代码，请稍候...")
    print("💡 提示：如果这是您第一次上传，电脑接下来会弹出 GitHub 网页登录授权窗口。")
    print("------------------------------------------------------------\n")

    if run_cmd("git push -u origin main"):
        print("\n🎉🎉🎉 【大功告成】！你的项目已经成功上传到 GitHub 啦！")
    else:
        print("\n❌ 推送失败，请检查网络或是否完成网页授权。")


if __name__ == "__main__":
    auto_push()