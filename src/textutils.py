import re
import math

class CompactPreprocessor:
    """
    Preprocessor that tries to reduce the number of tokens by removing punctuation
    """
    def convert(self, look, recipe, inventory, entities):
        if recipe == 'missing recipe' or recipe == '':
            txt = self.inventory_text(inventory, entities) + ' missing recipe ' + look
        elif recipe and type(recipe) == str:
            txt = self.inventory_text(inventory, entities) + ' ' + recipe + ' ' + look
        elif math.isnan(recipe):
            txt = self.inventory_text(inventory, entities) + ' missing recipe ' + look
        else:
            txt = self.inventory_text(inventory, entities) + ' ' + recipe + ' ' + look

        txt = re.sub(r'\n', ' ', txt)
        # convert names with hiffen with space
        txt = re.sub(r'(\w)\-(\w)', r'\1 \2', txt)
        # remove punctuation
        txt = re.sub(r'([.:\-!=#",?])', r' ', txt)
        txt = re.sub(r'\s{2,}', ' ', txt)
        return txt.strip('.')

    def inventory_text(self, inventory, entities):
        n_items = self.count_inventory_items(inventory, entities)
        text = '{} {}'.format(n_items, inventory)
        return text

    def count_inventory_items(self, inventory, entities):
        parts = [p.strip() for p in inventory.split('\n')]
        parts = [p for p in parts if p]
        return len([p for p in parts if any(p.find(ent) != -1 for ent in entities)])
