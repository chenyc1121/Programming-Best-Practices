aloha

## 学习总结

- git add 将内容放入暂存区；git commit 保存本地版本；git push 同步到远端。
- 分支指向提交，更新 main 不会自动更新 for_fun。
- 当一个分支的提交已包含在当前分支中，merge 会显示 Already up to date。
- 可以用 detached HEAD 查看旧版本，再用 git switch main 返回最新版本。
- MNIST 灰度图需要转为 RGB，再 resize 和归一化后输入模型。
- 本实验使用预训练骨干和未训练的 10 类分类头，10000 张测试图片准确率为 9.76%。
- 使用 .gitignore 排除数据集和模型权重，并记录依赖版本和实验结果。
