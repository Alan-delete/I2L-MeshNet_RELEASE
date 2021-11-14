#images
mkdir images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
mv val2017/ images/
mv train2017/ images/
# annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
