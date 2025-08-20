import math

from src.utils.testing import COLOR_MAP, RESET
from src.visualization.visualizer import DataVisualizer

class ModelTester:
    def __init__(self, predictor, title=None, data=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []
        self.visualizer = DataVisualizer()

    def color_for(self, error, truth):
        if error<40 or error/truth < 0.2:
            return "green"
        elif error<80 or error/truth < 0.4:
            return "orange"
        else:
            return "red"
    
    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        self.visualizer.create_prediction_scatter_plot(
            self.truths, 
            self.guesses, 
            self.colors, 
            title
        )

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color=="green")
        
        # Create base title
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        
        # Add optimization info if available
        if hasattr(self.predictor, 'optimization_info'):
            opt_info = self.predictor.optimization_info
            strategy = opt_info.get('strategy', 'Unknown')
            title += f"\nOptimized: {strategy}"
            
            # Different strategies have different metrics
            if 'best_seed' in opt_info:
                title += f" | Seed={opt_info['best_seed']}"
                if 'training_error' in opt_info:
                    title += f" | Train Error=${opt_info['training_error']:.2f}"
                if opt_info.get('validation_error'):
                    title += f" | Val Error=${opt_info['validation_error']:.2f}"
            elif 'training_r2' in opt_info:
                title += f" | Train R²={opt_info['training_r2']:.3f}"
                if opt_info.get('validation_r2'):
                    title += f" | Val R²={opt_info['validation_r2']:.3f}"
        
        self.chart(title)

    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function):
        cls(function).run()