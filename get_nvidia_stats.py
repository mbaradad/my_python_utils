import GPUtil
import time
import json

def get_gpu_stats(counts=10, desired_time_diffs_ms=0):
  gpus = [dict(gpu_usage=0, mem_usage=0, mem_total=0) for _ in GPUtil.getGPUs()]
  for _ in range(counts):
    t0 = time.time()
    for gpu_i, gpu in enumerate(GPUtil.getGPUs()):
      gpus[gpu_i]['gpu_usage'] += gpu.load
      gpus[gpu_i]['mem_usage'] += gpu.memoryUsed / gpu.memoryTotal
      time_diff_s = time.time() - t0
      if time_diff_s < desired_time_diffs_ms / 1000.0:
        time.sleep((desired_time_diffs_ms - time_diff_s) / 1000.0)
      t0 = time.time()


  for gpu_i, gpu in enumerate(GPUtil.getGPUs()):
    gpus[gpu_i]['gpu_usage'] /= counts
    gpus[gpu_i]['mem_usage'] /= counts
    gpus[gpu_i]['mem_total'] = gpu.memoryTotal

  return gpus


json_object = json.dumps(get_gpu_stats())
print(json_object)