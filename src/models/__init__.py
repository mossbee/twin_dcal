"""
DCAL Models Package
Main model implementations for Twin Face Verification
"""

from .dcal_verification_model import (
    DCALVerificationModel,
    VerificationHead,
    create_dcal_model
)

__all__ = [
    'DCALVerificationModel',
    'VerificationHead', 
    'create_dcal_model'
] 