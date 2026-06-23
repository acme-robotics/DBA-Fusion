import numpy as np, cv2, os, json
np.random.seed(0)
# Consistent VI control: camera sinusoidal translation viewing a planar wall at depth Z,
# IMU accel = exact 2nd-derivative of the metric camera translation, gyro=0, gravity on +z.
# cam frame == imu frame (Tbc=identity). Vision and inertial agree BY CONSTRUCTION.
root="_smokevi/mav0"; cam=f"{root}/cam0/data"; imu=f"{root}/imu0"
os.makedirs(cam,exist_ok=True); os.makedirs(imu,exist_ok=True)
W=900; canvas=(np.random.rand(W,W)*40).astype(np.uint8)
for _ in range(400):
    c=(np.random.randint(0,W),np.random.randint(0,W)); r=np.random.randint(4,28)
    cv2.circle(canvas,c,r,int(np.random.randint(60,255)),-1)
for _ in range(120):
    p1=(np.random.randint(0,W),np.random.randint(0,W)); p2=(np.random.randint(0,W),np.random.randint(0,W))
    cv2.line(canvas,p1,p2,int(np.random.randint(80,255)),2)
canvas=cv2.GaussianBlur(canvas,(3,3),0)

N=80; S=512; fps=20.0; dt=1.0/fps; f=250.0; cxcy=256.0; Z=4.0
D=(N-1)*dt
f1=0.25; f2=0.5                      # Hz, one and two periods over D=4s
Ax_px, Ay_px = 60.0, 20.0           # crop pixel amplitudes (as in the smoketest)
Ax_m = Ax_px*Z/f; Ay_m = Ay_px*Z/f  # metric camera translation amplitudes
def Px(t): return Ax_m*np.sin(2*np.pi*f1*t)
def Py(t): return Ay_m*np.sin(2*np.pi*f2*t)
def ax(t): return -Ax_m*(2*np.pi*f1)**2*np.sin(2*np.pi*f1*t)   # Px''
def ay(t): return -Ay_m*(2*np.pi*f2)**2*np.sin(2*np.pi*f2*t)   # Py''

t0=100_000_000_000
cam_csv=open(f"{root}/cam0/data.csv","w"); cam_csv.write("#t[ns],filename\n")
for i in range(N):
    tt=i*dt
    x=int(round(190+(f/Z)*Px(tt))); y=int(round(190+(f/Z)*Py(tt)))
    crop=canvas[y:y+S,x:x+S]
    img=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
    t=t0+int(tt*1e9); fn=f"{t}.png"
    cv2.imwrite(f"{cam}/{fn}",img); cam_csv.write(f"{t},{fn}\n")
cam_csv.close()

fimu=200.0; n_imu=int(N*dt*fimu)
imu_csv=open(f"{imu}/data.csv","w"); imu_csv.write("#t[ns],wx,wy,wz,ax,ay,az\n")
# demo divides gyro cols by deg->rad after a deg multiply in the loader; we emit rad/s=0 anyway.
for k in range(n_imu):
    tt=k/fimu
    t=t0+int(tt*1e9)
    imu_csv.write(f"{t},0.0,0.0,0.0,{ax(tt):.6f},{ay(tt):.6f},9.81\n")
imu_csv.close()

with open("_smokevi/calib.txt","w") as fcal: fcal.write(f"{f} {f} {cxcy} {cxcy}\n")
cfg={"source":"synthetic","image_size":[S,S],"intrinsics":[f,f,cxcy,cxcy],
     "distortion_model":"radtan","distortion":[0,0,0,0],
     "Tbc":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
     "imu_params":[1e-3,1e-4,1e-4,1e-5],"imu_rate":200.0,"timeshift_s":0.0}
json.dump(cfg,open("_smokevi/config.json","w"),indent=2)
print("consistent VI synth:",N,"frames",n_imu,"imu  metricAmp(x,y)=%.2f,%.2f m  accAmp(x,y)=%.2f,%.2f m/s^2"
      %(Ax_m,Ay_m,Ax_m*(2*np.pi*f1)**2,Ay_m*(2*np.pi*f2)**2))
