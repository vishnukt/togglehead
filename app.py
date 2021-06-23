
from flask import (
    Flask, render_template, request, redirect, flash, url_for, send_file
)
from werkzeug.utils import secure_filename
import os, sys, imageio, string, random
sys.path.insert(1, './first-order-model/')
from demo import load_checkpoints, make_animation
from skimage import img_as_ubyte
from skimage.transform import resize


UPLOAD_FOLDER = '/home/kt_vishnu19/toggle/togglehead/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.form.getlist('video')
        image = request.form.getlist('image')
        file = request.files['source_image']
        source_video = request.files['source_video']
        # check if the post request has the file part
        if (
            ((file.filename == '' or not allowed_image_file(file.filename)) and not image)
            or ((source_video.filename == '' or not allowed_video_file(source_video.filename)) and not video)
        ):
            flash('No selected file')
            return redirect(request.url)
            #return 'No File Submited'
        if not file.filename == '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = './static/uploads/'+filename
        else:
            filepath = './static/images/'+image[0]+'.png'
        if not source_video.filename == '':
            video_filename = secure_filename(source_video.filename)
            source_video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
            dvideo = './static/uploads/'+video_filename
        else:
            dvideo = './media/'+video[0]+'.mp4'

        source_image = imageio.imread(filepath)
        driving_video = imageio.mimread(dvideo, memtest=False)

        #Resize image and video to 256x256
        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        generator, kp_detector = load_checkpoints(config_path='./first-order-model/config/vox-256.yaml', 
            checkpoint_path='./media/vox-cpk.pth.tar', cpu=True)

        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, cpu=True)

        #save resulting video
        v_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))
        v_name = './static/output/'+v_name+'.mp4'
        imageio.mimsave(v_name, [img_as_ubyte(frame) for frame in predictions])
        # return send_file(v_name, mimetype='video')
        return render_template(
            'index.html',
            output_video = v_name,
            poster_image = filepath
        )
    # filepath = './static/images/emilia.png'
    return render_template('index.html')

@app.route('/toggle', methods=['GET', 'POST'])
def toggle():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No selected file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_image_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = './uploads/'+filename
            source_image = imageio.imread(filepath)
            dvideo = './media/rock.mp4'
            driving_video = imageio.mimread(dvideo, memtest=False)

            #Resize image and video to 256x256
            source_image = resize(source_image, (256, 256))[..., :3]
            driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

            generator, kp_detector = load_checkpoints(config_path='./first-order-model/config/vox-256.yaml', 
                checkpoint_path='./media/vox-cpk.pth.tar', cpu=True)

            predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, cpu=True)

            #save resulting video
            v_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))
            v_name = './'+v_name+'.mp4'
            imageio.mimsave(v_name, [img_as_ubyte(frame) for frame in predictions])

            return send_file(v_name, mimetype='video')
    return render_template('toggle.html')

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(host="0.0.0.0", port=8000, debug=True)