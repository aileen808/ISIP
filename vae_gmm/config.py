class Config(object):
  def __init__(self):
    self.learning_rate = 0.0001
    self.batch_size = 128
    self.win_size = 150
    self.n_traces = 2
    self.display_step = 50
    self.n_threads = 2
    #self.n_epochs = None
    self.epoch = 100
    self.regularization = 1e-3
    self.n_clusters = None

    # Number of epochs, None is infinite
    #n_epochs = None
