import numpy as np, cv2, os
np.random.seed(0)
root="_smoke/mav0"; cam=f"{root}/cam0/data"; imu=f"{root}/imu0"
os.makedirs(cam,exist_ok=True); os.makedirs(imu,exist_ok=True)
# big textured canvas
W=900; canvas=(np.random.rand(W,W)*40).astype(np.uint8)
for _ in range(400):
    c=(np.random.randint(0,W),np.random.randint(0,W)); r=np.random.randint(4,28)
    cv2.circle(canvas,c,r,int(np.random.randint(60,255)),-1)
for _ in range(120):
    p1=(np.random.randint(0,W),np.random.randint(0,W)); p2=(np.random.randint(0,W),np.random.randint(0,W))
    cv2.line(canvas,p1,p2,int(np.random.randint(80,255)),2)
canvas=cv2.GaussianBlur(canvas,(3,3),0)
N=80; S=512; fps=20.0; dt=1.0/fps
# sideways translation of crop window (sinusoid) -> optical flow
xs=190+60*np.sin(np.linspace(0,2*np.pi,N)); ys=190+20*np.sin(np.linspace(0,4*np.pi,N))
t0=100_000_000_000  # ns
cam_csv=open(f"{root}/cam0/data.csv","w"); cam_csv.write("#t[ns],filename\n")
for i in range(N):
    x,y=int(xs[i]),int(ys[i]); crop=canvas[y:y+S,x:x+S]
    img=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
    t=t0+int(i*dt*1e9); fn=f"{t}.png"
    cv2.imwrite(f"{cam}/{fn}",img); cam_csv.write(f"{t},{fn}\n")
cam_csv.close()
# IMU at 200Hz: gravity on z, small accel from the sideways motion (2nd deriv of xs), gyro~0
fimu=200.0; n_imu=int(N*dt*fimu)
imu_csv=open(f"{imu}/data.csv","w"); imu_csv.write("#t[ns],wx,wy,wz,ax,ay,az\n")
for k in range(n_imu):
    t=t0+int(k/fimu*1e9)
    imu_csv.write(f"{t},0.0,0.0,0.0,0.0,0.0,9.81\n")
imu_csv.close()
# 4-value pinhole calib (no fisheye) for ~512 crop
with open("_smoke/calib.txt","w") as f: f.write("250.0 250.0 256.0 256.0\n")
print("synthetic dataset:",N,"frames,",n_imu,"imu samples")
