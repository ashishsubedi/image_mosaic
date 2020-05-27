# Photo Mosaic

### Requirements

Python 3, OpenCV 4, Numpy, tqdm (<code>pip install tqdm </code>)

# How to Use

1. Clone or Download Repo
2. Run mosaic.py <code> python mosaic.py </code>

- <h3> Use <code> -i [path] </code> flag to add your content image
- <h3> Use <code> -d </code> to add you datasets of images
- <h3> Use <code> -r </code> to division of the image (Optional) . <b> Default is 32. Higher value leads to better picture but it takes more time. </b>
- <h3> Use <code> -s </code> to specify size of output (Optional)
- <h3> Use <code> -o [path] </code> to save image. (Add filename with path)

Example  
<code>python mosaic.py -i "C:\\me.jpg" -s 64 -d ./images -s (600,600) -o ./me.jpg </code>
