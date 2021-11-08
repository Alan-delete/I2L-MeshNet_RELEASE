#dwnload H36M annotations
#mkdir annotations
#cd annotations
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/h36m_annot.tar
#tar -xf h36m_annot.tar
#rm h36m_annot.tar
#cd ..

# Download H36M images
#mkdir -p images
cd images
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S1.tar
#tar -xf S1.tar
#rm S1.tar
#wget http://visiondata.cis.upenn.edu/volumetric/h36m/S5.tar
#tar -xf S5.tar
#rm S5.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S6.tar
tar -xf S6.tar
rm S6.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S7.tar
tar -xf S7.tar
rm S7.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S8.tar
tar -xf S8.tar
rm S8.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S9.tar
tar -xf S9.tar
rm S9.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S11.tar
tar -xf S11.tar
rm S11.tar
cd ../
