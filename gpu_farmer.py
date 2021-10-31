# author: muzhan
import os
import sys
import time
import threading
import argparse
import torch



def gpu_info(gpu_index=None):
    all_info = os.popen('nvidia-smi|grep %').read().strip().split('\n')
    if gpu_index is None:
        gpu_index = list(range(len(all_info)))
    elif type(gpu_index) == int:
        gpu_index = [gpu_index]

    gpu_stat = {}
    for idx in gpu_index:
        info = all_info[idx].split('|')
        max_power = int(info[1].split()[-1].strip('W'))
        max_memory = int(info[2].split('/')[1].strip().strip('MiB'))
        power = int(info[1].split()[-3].strip('W'))
        memory = int(info[2].split('/')[0].strip().strip('MiB'))
        gpu_stat[str(idx)]=(power, memory, max_power, max_memory)
    return gpu_stat


# TODO multi-thread version
# class gpu_monitor(threading.Thread):
#     def __init__(self, gpuID, available_list, memory_require, power_require):
#         threading.Thread.__init__(self)
#         self.gpu=gpuID
#         self.available_list=available_list
#
#     def run(self):
#
#
#         threadLock.acquire()
#         threadLock.release()



def narrow_setup(cmd, ngpu=1, memory_require=5000, power_require=-1,
                 gpu_index=None, interval=2, occupy=True):
    available_gpus=[]

    while len(available_gpus) < ngpu:
        gpu_stat = gpu_info(gpu_index)
        print_info='\r'
        for idx,(power, memory, max_power, max_memory) in list(gpu_stat.items()):
            symbol = f'monitoring gpu-{idx}: ' + '>' * 5 + ' ' * 3 + '|'
            gpu_power_str = f'gpu power:{power}W |'
            gpu_memory_str = f'gpu memory: {memory}MiB |'
            print_info +=  symbol + ' ' + gpu_memory_str + ' ' + gpu_power_str

            if idx in available_gpus:
                continue
            if int(max_memory * 0.95) - memory < memory_require or (max_power - power < power_require and power_require >= 0):  # set waiting condition
                continue
            available_gpus.append(idx)

            if occupy:
                block_mem = memory_require
                x = torch.FloatTensor(256, 1024, block_mem).to(device=torch.device(f'cuda:{idx}'))
                del x

        sys.stdout.write(f'\rCurrent available gpus: {available_gpus}!')
        sys.stdout.flush()

        if occupy == False and len(available_gpus) < ngpu:
            available_gpus = []

        time.sleep(interval)

    current_stat = gpu_info(gpu_index)
    free_mem_dict = {m:(current_stat[m][3] - current_stat[m][1]) for m in available_gpus}
    available_gpus.sort(key=lambda x: free_mem_dict[x], reverse=True)
    prefix = 'CUDA_VISIBLE_DEVICES=\"' + ','.join(available_gpus) + '\" '
    print('\n' + prefix + cmd)
    torch.cuda.empty_cache()
    os.system(prefix + cmd)
    #return available_gpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cmd", help="The program to execute;",
                        type=str, required=True)
    parser.add_argument("-n", "--ngpu", help="How many gpus you need;",
                        type=int, default=1)
    parser.add_argument("-m", "--memory_require", help="For each gpu, how much free memory is needed at least;",
                        type=int, default=5000)
    parser.add_argument("-p", "--power_require", help="For each gpu, how much power capacity is needed at least; -1 means no power limit;",
                        type=int, default=-1)
    parser.add_argument("--gpu_index", help="Choose gpus from following device ids, default: all;",
                        type=list)
    parser.add_argument("--interval", help="Scan once every ? seconds;",
                        type=int, default=2)
    parser.add_argument("-f", "--occupy", help="whether to occupy the memory, default: False",
                        action="store_true")
    args=parser.parse_args()
    print(args)
    narrow_setup(**vars(args))
    #exit(','.join(available_gpus))
