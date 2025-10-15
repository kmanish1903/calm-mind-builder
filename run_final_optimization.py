#!/usr/bin/env python3
"""
Final Optimization Runner
Execute all Phase 9 optimization and testing procedures
"""

import sys
import subprocess
from pathlib import Path
from optimize_production import ProductionOptimizer
from testing.test_suite import TestSuite
from testing.security_audit import SecurityAuditor
from testing.performance_optimizer import PerformanceOptimizer

def run_comprehensive_optimization():
    """Run complete optimization pipeline"""
    print("🚀 Starting Comprehensive Production Optimization")
    print("=" * 60)
    
    # Initialize optimizers
    prod_optimizer = ProductionOptimizer()
    test_suite = TestSuite()
    security_auditor = SecurityAuditor()
    perf_optimizer = PerformanceOptimizer()
    
    # Step 1: Run test suite
    print("\n📋 Step 1: Running Test Suite...")
    try:
        test_suite.run_all_tests()
        print("✅ Test suite completed")
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
    
    # Step 2: Security audit
    print("\n🔒 Step 2: Security Audit...")
    try:
        security_auditor.run_comprehensive_audit()
        print("✅ Security audit completed")
    except Exception as e:
        print(f"❌ Security audit failed: {e}")
    
    # Step 3: Performance optimization
    print("\n⚡ Step 3: Performance Optimization...")
    try:
        perf_optimizer.optimize_application()
        print("✅ Performance optimization completed")
    except Exception as e:
        print(f"❌ Performance optimization failed: {e}")
    
    # Step 4: Production optimization
    print("\n🏭 Step 4: Production Optimization...")
    try:
        prod_optimizer.run_full_optimization()
        print("✅ Production optimization completed")
    except Exception as e:
        print(f"❌ Production optimization failed: {e}")
    
    # Step 5: Final build test
    print("\n🔨 Step 5: Final Build Test...")
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print("✅ Production build successful")
        else:
            print(f"❌ Production build failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Build test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive optimization completed!")
    print("\n📊 Generated Reports:")
    print("- OPTIMIZATION_REPORT.md")
    print("- TEST_RESULTS.md")
    print("- SECURITY_AUDIT.md")
    print("- PERFORMANCE_REPORT.md")
    
    print("\n📋 Next Steps:")
    print("1. Review all generated reports")
    print("2. Address any critical issues found")
    print("3. Run final deployment checks")
    print("4. Deploy to production environment")

if __name__ == "__main__":
    run_comprehensive_optimization()