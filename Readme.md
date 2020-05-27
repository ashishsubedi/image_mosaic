# Photo Mosaic

### Requirements

Python 3, OpenCV 4, Numpy, tqdm (<code>pip install tqdm </code>)

# How to Use

1. Clone or Download Repo

2. Open CMD in the repo location and run mosaic.py <code> python mosaic.py </code> (https://www.thewindowsclub.com/how-to-open-command-prompt-from-right-click-menu/)

- <h2> Note: If you want the above command to work, copy your image into the folder and rename it to me.jpg and add other image samples in the folder called images</h2>
- <h3> Use <code> -i [path] </code> flag to add your content image
- <h3> Use <code> -d </code> to add you datasets of images
- <h3> Use <code> -r </code> to division of the image (Optional) . <b> Default is 32. Higher value leads to better picture but it takes more time. </b>
- <h3> Use <code> -s </code> to specify size of output (Optional)
- <h3> Use <code> -o [path] </code> to save image. (Add filename with path)

<h1> Example </h1>
<code>python mosaic.py -i me.jpg -r 64 -d ./images -s 600 600 -o ./output.jpg </code>
