Usage
=====

.. _installation:

Installation
------------

To use CSST Emulator, you just need get the source code from GitHub for now:

.. code-block:: console

   (.venv) $ git clone https://github.com/czymh/csstemu.git

Creating recipes
----------------

To use the package, you need add the code directory to your Python path:
>>> import sys
>>> sys.path.append('path/to/csstemu')
>>> from CEmulator.Emulator import CEmulator

Firstly, you need to create an object of the class :py:class:`CEmulator`:
>>> csstemu = CEmulator(verbose=True)
Then set the cosmologies you want to use:
>>> csstemu.set_cosmos(Omegab=Omegab, Omegam=Omegam, H0=h0*100,
>>>            ns=n_s, As=A_s, w=w0, 
>>>            wa=wa, mnu=m_nu)
All these variables can be float numbers or arrays.
Finally, you can predict the matter power spectrum:
>>> csstemu.get_pknl(z=zlist, k=klist, Pcb=True, lintype='Emulator', nltype='linear')

.. autofunction:: CEmulator.get_pknl
``z`` parameter is the redshift,
``k`` is the wavenumber with unit of :math:`h/\mathrm{Mpc}`,
``Pcb`` is the flag to return power spectrum of cold dark matter+baryon (default is True),
else return the total matter power spectrum includes the massive neutrinos.  
``lintype`` is the linear power spectrum type, which should be 'Emulator' or 'CLASS' 
``nltype``  determine the type of emulation: the 'linear' or 'halofit'.
