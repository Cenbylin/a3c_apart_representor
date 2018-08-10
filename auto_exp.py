# 日志
import logging
import sys
import os
import time
logger = logging.getLogger("exp")
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler("auto_exp.log", mode="w", encoding="UTF-8")
file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
console_handler = logging.StreamHandler(sys.stdout)# 控制台
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

actor_weight = [0.0, 0.2, 0.4]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

counter = 1
for aw in actor_weight:
        logger.info("exp-%s for aw(%s) time:%s", str(counter), str(aw), str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        os.system("python main.py --env SpaceInvaders-v0 --workers 6 --gpu-ids 0 --amsgrad True --pre-rnet human_1env_novae --log-target human_1env_novae_aw{} --actor-weight {} --max-step 400000".format(aw, aw))
        counter += 1
        time.sleep(2)
        
logger.info("finished!")
#nohup python main.py --env SpaceInvaders-v0 --workers 6 --gpu-ids 1 1 --amsgrad True  --log-target a3c --max-step 2000000 &