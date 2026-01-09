import pandas as pd


from density_est import DensityKernel, DensityNeighbor


class DensityBalance():
    def __init__(self, model='Kernel'):
        models = {'Neighbor': DensityNeighbor(), "Kernel": DensityKernel()}
        self.model = models[model]

    def weight(self, A, y, treatment="Reweighing"):
        dy = pd.DataFrame(y,columns=["y"])
        dA = pd.DataFrame(A, columns=["A"])
        X = pd.concat((dA, dy), axis=1)

        w = self.model.density(X)
        wA = self.model.density(dA)
        wy = self.model.density(dy)

        w = (len(w) * w / sum(w)).flatten()
        wA = (len(wA) * wA / sum(wA)).flatten()
        wy = (len(wy) * wy / sum(wy)).flatten()


        if treatment == "FairBalanceVariant":
            weight = 1 / w
        elif treatment == "FairBalance":
            weight = wA / w
        elif treatment == "GroupBalance":
            weight = wy / w
        elif treatment == "Reweighing":
            weight = wA * wy / w


        weight = (len(weight) * weight / sum(weight)).flatten()
        return weight