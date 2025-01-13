Quick Usage
===========

.. _installation:

Installation
------------

To use CSST Emulator, you can install it via pip:

.. code-block:: console

   $ pip install git+https://github.com/czymh/csstemu

for the stable version, or

.. code-block:: console

   $ pip install git+https://github.com/czymh/csstemu/tree/dev

for the development version to use the latest function.

Creating recipes
----------------

To use the package, you need add the code directory to your Python path:

.. code-block:: python

   from CEmulator.Emulator import CEmulator

Firstly, you need to create an object of the class :py:class:`CEmulator`:

.. code-block:: python

   csstemu = CEmulator(verbose=True)

Then set the cosmologies you want to use:

.. code-block:: python

   csstemu.set_cosmos(Omegab=Omegab, Omegac=Omegac, H0=h0*100,
                      ns=n_s, As=A_s, w=w0, 
                      wa=wa, mnu=m_nu)

All these variables can be float numbers or arrays.
Finally, you can predict the matter power spectrum:

.. code-block:: python

   csstemu.get_pknl(z=zlist, k=klist, Pcb=False, lintype='Emulator', nltype='hmcode2020')

.. autofunction:: CEmulator.Emulator.CEmulator.get_pknl
   :no-index:
