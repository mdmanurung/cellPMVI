Fusion Strategies
=================

cellPMVI supports three strategies for combining information from
RNA and protein modalities in the latent space.

Product-of-Experts (PoE)
------------------------

The default strategy. Combines modality-specific Gaussian posteriors by
multiplying them:

.. math::

   \frac{1}{\sigma^2_{\text{joint}}} = \frac{1}{\sigma^2_{\text{prior}}} + \sum_i \frac{1}{\sigma^2_i}

.. math::

   \mu_{\text{joint}} = \sigma^2_{\text{joint}} \left( \frac{\mu_{\text{prior}}}{\sigma^2_{\text{prior}}} + \sum_i \frac{\mu_i}{\sigma^2_i} \right)

**Properties:**

- Produces a **sharper** (more confident) joint posterior
- Each modality contributes proportionally to its precision (1/variance)
- Includes a standard normal prior as a regularizer

.. code-block:: python

   model = CellPMVI(adata, fusion_method="poe")

Mixture-of-Experts (MoE)
-------------------------

Averages the modality-specific posteriors:

.. math::

   \mu_{\text{joint}} = \frac{1}{M} \sum_i \mu_i

.. math::

   \sigma^2_{\text{joint}} = \frac{1}{M} \sum_i (\sigma^2_i + \mu_i^2) - \mu_{\text{joint}}^2

**Properties:**

- Produces a **broader** joint posterior
- More robust to missing or noisy modalities
- No single modality can dominate the posterior

.. code-block:: python

   model = CellPMVI(adata, fusion_method="moe")

None (cross-modal reconstruction)
----------------------------------

No posterior fusion. Each modality maintains its own independent latent
representation, but cross-modal reconstruction is performed using a
2x2 reconstruction matrix:

.. list-table::
   :header-rows: 1

   * -
     - RNA decoder
     - Protein decoder
   * - RNA encoder
     - RNA → RNA
     - RNA → Protein
   * - Protein encoder
     - Protein → RNA
     - Protein → Protein

This encourages the latent spaces to capture shared biological variation
without explicitly fusing them.

.. code-block:: python

   model = CellPMVI(adata, fusion_method="none")

Choosing a strategy
-------------------

.. list-table::
   :header-rows: 1

   * - Strategy
     - Best for
     - Trade-off
   * - **PoE**
     - Complete multi-modal data
     - Sharpest posteriors, but sensitive to modality quality
   * - **MoE**
     - Data with variable modality quality
     - More robust, but broader posteriors
   * - **None**
     - Exploratory analysis
     - Preserves modality-specific structure
