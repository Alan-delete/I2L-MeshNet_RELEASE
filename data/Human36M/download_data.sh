#dwnload H36M annotations
#mkdir annotations
#cd annotations
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/h36m_annot.tar
#tar -xf h36m_annot.tar
#rm h36m_annot.tar
#cd ..
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ztokDig-Ayi8EYipGE1lchg5XlAoLmwY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ztokDig-Ayi8EYipGE1lchg5XlAoLmwY" -O annotations.zip && rm -rf /tmp/cookies.txt
#unzip annotations.zip
#rm annotations.zip
# Download H36M images
#mkdir -p images
#cd images
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S1.tar
#tar -xf S1.tar
#rm S1.tar
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S5.tar
#tar -xf S5.tar
#rm S5.tar
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S6.tar
#tar -xf S6.tar
#rm S6.tar
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S7.tar
#tar -xf S7.tar
#rm S7.tar
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S8.tar
#tar -xf S8.tar
#rm S8.tar
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S9.tar
#tar -xf S9.tar
#rm S9.tar
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S11.tar
#tar -xf S11.tar
#rm S11.tar
#cd ../


#images.tar.gzaa
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A1SvZo4poulpuN0Nz9nOxogiy5b4J5Lo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A1SvZo4poulpuN0Nz9nOxogiy5b4J5Lo" && rm -rf /tmp/cookies.txt
#images.tar.gzab
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UgSocwxevJh4s6C47Fmzvy5t6T3YOOnV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UgSocwxevJh4s6C47Fmzvy5t6T3YOOnV" && rm -rf /tmp/cookies.txt
#images.tar.gzac
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18czfW6irg909UV3FGgc-BGfAHyyS3ssq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18czfW6irg909UV3FGgc-BGfAHyyS3ssq" && rm -rf /tmp/cookies.txt
#images.tar.gzad
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NmshE-mdCeNTmkJS_bHhj8uii84ZKQrM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NmshE-mdCeNTmkJS_bHhj8uii84ZKQrM" && rm -rf /tmp/cookies.txt
#images.tar.gzae
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11TuzJ2CF_k3GdWRXadADwMDYEjrbe54u' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11TuzJ2CF_k3GdWRXadADwMDYEjrbe54u" && rm -rf /tmp/cookies.txt
#images.tar.gzaf
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZUcjfVmHjx-VO29oiTDoFUu9RLXTcnxY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZUcjfVmHjx-VO29oiTDoFUu9RLXTcnxY" && rm -rf /tmp/cookies.txt
#images.tar.gzag
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AHGriYMrQavmjD-r4zI6HK7bSsGbM_Lr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AHGriYMrQavmjD-r4zI6HK7bSsGbM_Lr" && rm -rf /tmp/cookies.txt

cat images.tar.gz* | tar -zxvpf -
