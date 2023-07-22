import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

def display_image(image_file_path):
    # Create a GStreamer pipeline
    pipeline = Gst.parse_launch(f'playbin uri=file://{image_file_path}')

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Run the GMainLoop (main event loop)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    # Stop the pipeline and clean up
    pipeline.set_state(Gst.State.NULL)
