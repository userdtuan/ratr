# Import the Flask class from the flask module
from flask import Flask

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the root URL ("/") - This is the default route
@app.route('/')
def hello_world():
    return 'Hello, World! This is a simple Flask app.'

# Run the application if this script is executed
if __name__ == '__main__':
    app.run(debug=True)
