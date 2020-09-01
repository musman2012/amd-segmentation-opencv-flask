import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import matplotlib.pyplot as plt
#from werkzeug import secure_filename

import testcv
# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set([ 'png', 'jpg', 'jpeg', 'JPG', 'tif', 'bmp'])

app.config['my_str'] = ''

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    print request
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        #filename = secure_filename(file.filename)
        filename = file.filename
        # Move the file form the temporal folder to
        # the upload folder we setup
        dir_path = '/app/uploads/'
        path = os.path.join(dir_path, filename)
        file.save(path)
        #plt.savefig(file)
        #plt.show()
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        print filename
        res_str = testcv.image_make(path, filename)
        #my_str += res_str
        app.config['my_str'] = res_str


        #return render_template('result.html')
        #return redirect(url_for('uploaded_file',
        #                         filename=path[5:]))
        #return send_from_directory(dir_path, filename)
        return redirect(url_for('uploaded_file'))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/templates/result.html')
def uploaded_file():
    #return send_from_directory(app.config['UPLOAD_FOLDER'],
    #                           filename)
    #return send_from_directory(filename)
    return render_template('result.html', return_str = app.config['my_str'])

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':
    app.run(
        '0.0.0.0', debug=True
    )
