# 定义重启延迟时间（秒）
RESTART_DELAY=1

echo "Supervisor started at $(date)" | tee -a log/supervisor.log

while true; do
    echo "Starting train.py at $(date)" | tee -a log/supervisor.log
    
    # 运行 train.py, 并捕获其退出状态码
    /root/miniconda3/bin/python -u train.py >> log/train.log 2>&1
    
    EXIT_CODE=$? # 获取上一个命令的退出状态码
    echo "train.py exited with code $EXIT_CODE at $(date)" | tee -a log/supervisor.log

    if [ $EXIT_CODE -eq 0 ]; then
        echo "train.py finished gracefully. Exiting supervisor." | tee -a log/supervisor.log
        break # 正常退出, 停止循环
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "Anomaly detected, train.py signaled restart. Waiting ${RESTART_DELAY}s before restarting..." | tee -a log/supervisor.log
        sleep ${RESTART_DELAY} # 异常退出, 等待一段时间后重启
    else
        echo "train.py exited with unexpected code $EXIT_CODE. Exiting supervisor." | tee -a log/supervisor.log
        break # 其他非零退出码, 停止循环
    fi
done

