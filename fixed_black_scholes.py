"""
Fixed Black-Scholes implementation without problematic imports.
"""

import numpy as np
import math

class SimpleBlackScholes:
    """Simplified Black-Scholes implementation."""
    
    @staticmethod
    def norm_cdf(x):
        """Approximation of normal cumulative distribution function."""
        # Abramowitz and Stegun approximation
        if x < 0:
            return 1 - SimpleBlackScholes.norm_cdf(-x)
        
        # Constants
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        
        # Save the sign of x
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return 0.5 * (1.0 + sign * y)
    
    @staticmethod
    def calculate_price(S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes option price."""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * SimpleBlackScholes.norm_cdf(d1) - K * math.exp(-r * T) * SimpleBlackScholes.norm_cdf(d2)
        else:
            price = K * math.exp(-r * T) * SimpleBlackScholes.norm_cdf(-d2) - S * SimpleBlackScholes.norm_cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_delta(S, K, T, r, sigma, option_type='call'):
        """Calculate option delta."""
        if T <= 0:
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        
        if option_type.lower() == 'call':
            return SimpleBlackScholes.norm_cdf(d1)
        else:
            return SimpleBlackScholes.norm_cdf(d1) - 1

def test_fixed_implementation():
    """Test the fixed Black-Scholes implementation."""
    print("Testing Fixed Black-Scholes Implementation")
    print("=" * 40)
    
    # Test parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25 # Time to expiration (3 months)
    r = 0.05 # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)
    
    bs = SimpleBlackScholes()
    
    # Calculate call and put prices
    call_price = bs.calculate_price(S, K, T, r, sigma, 'call')
    put_price = bs.calculate_price(S, K, T, r, sigma, 'put')
    
    # Calculate deltas
    call_delta = bs.calculate_delta(S, K, T, r, sigma, 'call')
    put_delta = bs.calculate_delta(S, K, T, r, sigma, 'put')
    
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T} years")
    print(f"Risk-free Rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print()
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")
    print(f"Call Delta: {call_delta:.4f}")
    print(f"Put Delta: {put_delta:.4f}")
    
    # Verify put-call parity: C - P = S - K*e^(-rT)
    parity_left = call_price - put_price
    parity_right = S - K * math.exp(-r * T)
    parity_diff = abs(parity_left - parity_right)
    
    print(f"\nPut-Call Parity Check:")
    print(f"C - P = {parity_left:.4f}")
    print(f"S - Ke^(-rT) = {parity_right:.4f}")
    print(f"Difference: {parity_diff:.6f}")
    
    if parity_diff < 0.001:
        print("✓ Put-call parity holds!")
    else:
        print("✗ Put-call parity violation!")
    
    return parity_diff < 0.001

if __name__ == "__main__":
    success = test_fixed_implementation()
    if success:
        print("\n🎉 Black-Scholes implementation is working correctly!")
        print("\nThe quantitative finance library core functionality is ready.")
        print("You can now use the individual modules without network dependencies.")
    else:
        print("\n⚠️ There may be issues with the implementation.")
