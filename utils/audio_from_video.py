import os
from moviepy import VideoFileClip

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v")

vid_base_dir = "abaw_dataset/video"
sub_path = ["val", "new_vids", "train"]
output_dir = "abaw_dataset/audio"

def extract_audio(sub_name, output_format=".wav"):
    
    try:
        video_path = os.path.join(vid_base_dir, sub_name)
        
        base_name = os.path.splitext(sub_name)[0]
        output_name = f"{base_name}{output_format}"
        output_path = os.path.join(output_dir, output_name)
       
        video_clip = VideoFileClip(video_path)

        video_clip.audio.write_audiofile(
            output_path, fps=16000, logger="bar"
        )

        video_clip.close()
        print(f"[成功] 保存至: {output_path}")

    except Exception as e:
        print(f"[错误] 处理 {video_path} 失败: {e}")


def batch_process_folder(sub_path):
    
    folder_path = os.path.join(vid_base_dir, sub_path)
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    files = os.listdir(folder_path)
    count = 0

    for file in files:
        
        full_path = os.path.join(folder_path, file)
        sub_name = os.path.join(sub_path, file)
        
        if os.path.isfile(full_path) and file.lower().endswith(SUPPORTED_EXTENSIONS):
            extract_audio(sub_name=sub_name)
            count += 1

    print(f"--- 处理完成，共处理了 {count} 个视频文件 ---")

for item in sub_path:
    batch_process_folder(sub_path=item)
