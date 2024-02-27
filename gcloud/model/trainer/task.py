"""Run a training job on Cloud ML Engine for a given use case.
Usage:
  trainer.task --train_data_path <train_data_path> --output_dir <outdir> --signal_variables <signal_variables> --control_variables <control_variables>
              [--batch_size <batch_size>] [--xl <xl>] [--ul <ul>] [--epochs <epochs>] [--normalization <normalization>]
Options:
  -h --help     Show this screen.
  --xl <xl>  Integer value indicating the number of signal variables [default: 8]
  --ul <ul>  Integer value indicating the number of control variables [default: 16]
  --label <label>  String value indicating the label column name [default: label]
  --batch_size <batch_size>  Integer value indiciating batch size [default: 64]
  --epochs <epochs>  Integer value indiciating the number of epochs [default: 100]
  --normalization <normalization>  String value indicating the type of normalization [default: standard]
  --s_dim <s_dim>  Integer value indicating the number of signal variables [default: 4]
  --s_activation <s_activation>  String value indicating the activation function for signal variables [default: tanh]
  --validation_split <validation_split>  Float value indicating the validation split [default: 0.1]
  --verbose <verbose>  Integer value indicating the verbosity of the model [default: 2]
"""
from docopt import docopt

from trainer.model import Model

if __name__ == '__main__':
    arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    Model.xl = int(arguments['--xl'])
    Model.ul = int(arguments['--ul']) or Model.xl * 2
    Model.label = str(arguments['--label'])
    Model.epochs = int(arguments['--epochs'])
    Model.batch_size = int(arguments['--batch_size'])
    Model.normalization = str(arguments['--normalization'])
    Model.output_dir = str(arguments['<outdir>'])
    Model.signal_variables = arguments['--signal_variables'].split(',')
    Model.control_variables = arguments['--control_variables'].split(',')
    Model.s_dim = int(arguments['--s_dim'])
    Model.s_activation = str(arguments['--s_activation'])
    Model.validation_split = float(arguments['--validation_split'])
    Model.verbose = int(arguments['--verbose'])

    model = Model()

    # Load the data
    x_train, x_val = model.prepare_train_data(
      path=str(arguments['--train_data_path'])
    )

    # Train/Load the model
    # if str(arguments['--train_model_path']):
    #   model = model.load_model(
    #     path=str(arguments['--train_model_path'])
    #   )
    
    #   model.estimate_system(x_val)
    # else:

    # Run the training job, also saves the model
    model.train_model_with_estimation(x_train, x_val)

    # Testing
    # x_test, u_test, labels = model.prepare_test_data(
    #   path=str(arguments['--test_data_path'])
    # )  

    # model.test_and_evaluate(x_test, u_test, labels)