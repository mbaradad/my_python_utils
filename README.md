# Install  
To install the requirements and opencv + ffmpeg from conda, with:

```
pip install -r requirements
# if opencv gives errors (libstdc++.so.6: version `GLIBCXX_3.4.20' ...), install gcc first:
conda install libgcc
conda install -c conda-forge opencv
conda install -c conda-forge ffmpeg
```

For extra requirements (that are more strange and may be hard to find in some systems, like company clusters) install requirements_extra.txt. 
Without it some of the functionalities may fail, though they are rarely used in my projects:

`pip install -r requirements_extra.txt`