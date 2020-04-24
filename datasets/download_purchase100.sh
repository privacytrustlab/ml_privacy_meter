url="https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz"
wget ${url}
tar -xvzf dataset_purchase.tgz
python2 preprocess_purchase100.py
rm dataset_purchase.tgz dataset_purchase
