import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative imports
from lida.components.summarizer import Summarizer
from lida.components import viz, goal, scaffold, executor, manager, persona

# Optionally, you can explicitly list what should be imported when using 'from components import *'
__all__ = ['Summarizer', 'viz', 'goal', 'scaffold', 'executor', 'manager', 'persona']

# Explicitly import the Summarizer class
from .summarizer import Summarizer

# Keep the other imports as they are
from .viz import *
from .goal import *
from .scaffold import *
from .executor import *
from .manager import *
from .persona import *

# Optionally, you can explicitly list what should be imported when using 'from components import *'
__all__ = ['Summarizer', 'viz', 'goal', 'scaffold', 'executor', 'manager', 'persona']
