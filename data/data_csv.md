运行 manifest 构建或数据探测命令前，先进入统一 Python 环境：

```bash
source /share/project/liujingyi/activate_conda.sh
conda activate waverae
```

以下路径仅描述源数据 CSV，不代表 codec sequence 的起点。训练时随机采样出的每个完整
clip 都会被显式声明为 `sequence_origin=sampled_segment`，其首帧是该 codec segment
的 start，不要求对应源视频时间 0。

/home/shuchenweng/zhj/data/MTV_data_combine_movies_cdm_moviebench_sf20k/merged_without_sf20k_speaker=1_speaker_time_filter_speaker_time_higher_0.7_dialog_PQ_higher_6.5.csv
这个是单人说话的数据


/home/shuchenweng/zhj/data/MTV_data_combine_movies_cdm_moviebench_sf20k/MTV_effect_vocal=0_audio_prob_higher_0.8.csv
这个是 无人说话一定有环境音（脚步，汽车声等等）的


/home/shuchenweng/zhj/data/MTV_data_combine_movies_cdm_moviebench_sf20k/MTV_music_vocal=0.csv
这个是 无人说话一定有音乐的
