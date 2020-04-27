Prediction of Molecular Bioactivity for Drug Design -- Binding to Thrombin
--------------------------------------------------------------------------

Drugs are typically small organic molecules that achieve their desired
activity by binding to a target site on a receptor. The first step in
the discovery of a new drug is usually to identify and isolate the
receptor to which it should bind, followed by testing many small
molecules for their ability to bind to the target site. This leaves
researchers with the task of determining what separates the active
(binding) compounds from the inactive (non-binding) ones.  Such a
determination can then be used in the design of new compounds that not
only bind, but also have all the other properties required for a drug
(solubility, oral absorption, lack of side effects, appropriate duration
of action, toxicity, etc.). 

The present training data set consists of 1909 compounds tested for
their ability to bind to a target site on thrombin, a key receptor in
blood clotting. The chemical structures of these compounds are not
necessary for our analysis and are not included. Of these compounds, 42
are active (bind well) and the others are inactive. Each compound is
described by a single feature vector comprised of a class value (A for
active, I for inactive) and 139,351 binary features, which describe
three-dimensional properties of the molecule. The definitions of the
individual bits are not included - we don't know what each individual
bit means, only that they are generated in an internally consistent
manner for all 1909 compounds. Biological activity in general, and
receptor binding affinity in particular, correlate with various
structural and physical properties of small organic molecules. The task
is to determine which of these properties are critical in this case and
to learn to accurately predict the class value.  To simulate the
real-world drug design environment, the test set contains 636 additional
compounds that were in fact generated based on the assay results
recorded for the training set. In evaluating the accuracy, a
differential cost model will be used, so that the sum of the costs of
the actives will be equal to the sum of the costs of the inactives.

We thank DuPont Pharmaceuticals for graciously providing this data set
for the KDD Cup 2001 competition.  All publications referring to
analysis of this data set should acknowledge DuPont Pharmaceuticals
Research Laboratories and KDD Cup 2001.
