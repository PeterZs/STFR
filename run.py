import os
import argparse
import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--video_path', type=str, default="workspace/0227.mp4")
parser.add_argument('--video_step_size', default=10, type=int)
parser.add_argument('--video_ds_ratio', default=0.5, type=float)

# parser.add_argument('--reg_ldm_type', type=str, default="eyelid", choices=["ibug", "eyelid", "interp"])
parser.add_argument('--reg_close_eye', type=int, default=0)

parser.add_argument('--save_root', type=str, default="workspace/0227")

parser.add_argument('--func', type=str, default="extract-mat-recon-refine-reg-tex")


opt, _ = parser.parse_known_args()
opt.save_root = os.path.abspath(opt.save_root)


os.makedirs(opt.save_root, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(opt.save_root, f"log_{timestamp}.txt")


def write_log(message):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


start_all = datetime.datetime.now()
write_log(f"Start Job: {opt.func}")
write_log(f"Start Time: {start_all.strftime('%Y-%m-%d %H:%M:%S')}")
write_log("-" * 30)


code_root = os.path.dirname(os.path.abspath(__file__))
mat_code_root = os.path.join(code_root, "matting")
recon_code_root = os.path.join(code_root, "reconstruction")
refine_code_root = os.path.join(code_root, "refinement")
reg_code_root = os.path.join(code_root, "registration")
tex_code_root = os.path.join(code_root, "texture")

# set up save root
raw_frame_root = os.path.join(opt.save_root, "raw_frames")
mask_save_root = os.path.join(opt.save_root, "mask")


if "extract" in opt.func:
    m_start = datetime.datetime.now()
    os.makedirs(raw_frame_root, exist_ok=True)
    extract_frame_cmd = """
        ffmpeg -i %s \\
            -vf "select=not(mod(n\,%d)),scale=iw*%f:ih*%f,setsar=1:1" \\
            -vsync vfr -q:v 1 %s/%%05d.png
    """
    os.system(extract_frame_cmd % (opt.video_path, opt.video_step_size, opt.video_ds_ratio, opt.video_ds_ratio, raw_frame_root))
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: extract] runtime: {m_end - m_start}")

if "mat" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(mat_code_root)
    os.system("python run_matting.py --input_root %s --output_root %s" % (raw_frame_root, mask_save_root))
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: mat] runtime: {m_end - m_start}")

if "recon" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(recon_code_root)
    os.system("python run_reconstruction.py --data_root %s" % opt.save_root)
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: recon] runtime: {m_end - m_start}")

if "refine" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(refine_code_root)
    os.system("python run_refinement.py --data_root %s" % opt.save_root)
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: refine] runtime: {m_end - m_start}")

if "reg" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(reg_code_root)
    os.system("python run_registration.py --data_root %s --close_eye %d" % (opt.save_root, opt.reg_close_eye))
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: reg] runtime: {m_end - m_start}")

if "tex" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(tex_code_root)
    os.system("python run_texture.py --data_root %s" % os.path.join(opt.save_root, "sample_dataset"))
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: tex] runtime: {m_end - m_start}")

end_all = datetime.datetime.now()
write_log("-" * 30)
write_log(f"total runtime: {end_all - start_all}")
write_log(f"end time: {end_all.strftime('%Y-%m-%d %H:%M:%S')}")
