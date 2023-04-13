from models import *

GWO_controller = GWOController()
GWO_controller.use_model_1()
GWO_controller.run(times = 100)
GWO_controller.show()