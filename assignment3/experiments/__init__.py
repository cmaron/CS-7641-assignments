import datetime
import warnings

from tempfile import mkdtemp

import sklearn
import sklearn.model_selection as ms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve

from .base import *
from .benchmark import *
from .clustering import *
from .ICA import *
from .PCA import *
from .LDA import *
from .SVD import *
from .RF import *
from .RP import *
from .plotting import *
from .scoring import *

__all__ = ['pipeline_memory', 'run_subexperiment', 'clustering', 'benchmark', 'ICA', 'PCA', 'LDA', 'SVD', 'RF', 'RP']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# TODO: Fix this by changing the datatypes of the columns to float64?
warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)

warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

# Keep a cache for the pipelines to speed things up
pipeline_cachedir = mkdtemp()
# pipeline_memory = Memory(cachedir=pipeline_cachedir, verbose=10)
pipeline_memory = None


def run_subexperiment(main_experiment, out, ds=None):
    if not os.path.exists(out):
        os.makedirs(out)

    out = out + '/{}'
    details = main_experiment.get_details()
    # Run the clustering again as a sub-experiment for this one
    clustering_details = ExperimentDetails(
        details.ds if not ds else ds,
        details.ds_name,
        details.ds_readable_name,
        details.best_nn_params,
        details.threads,
        details.seed)
    ce = clustering.ClusteringExperiment(clustering_details, verbose=main_experiment.get_vebose())

    return ce.perform_for_subexperiment(out, main_experiment)
