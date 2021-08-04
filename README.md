# signature_recongnition
sudo jupyter notebook --no-browser --ip 0.0.0.0 --port 8999 --allow-root
ssh -N -f -L localhost:8999:localhost:8999 lyx@192.168.2.2 
ssh -N -f -L localhost:5000:localhost:5000 lyx@192.168.2.2 