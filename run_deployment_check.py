#!/usr/bin/env python3
"""
Deployment Check Runner
Execute comprehensive deployment readiness validation
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from deployment_readiness import DeploymentReadinessChecker
from testing.test_suite import TestSuite
from testing.security_audit import SecurityAuditor
from testing.performance_optimizer import PerformanceOptimizer

def run_complete_deployment_check():
    """Run complete deployment readiness check"""
    print("🚀 Starting Complete Deployment Readiness Check")
    print("="*60)
    
    # Initialize checkers
    readiness_checker = DeploymentReadinessChecker()
    test_suite = TestSuite()
    security_auditor = SecurityAuditor()
    performance_optimizer = PerformanceOptimizer()
    
    results = {}
    
    try:
        # 1. Run deployment readiness check
        print("\n📋 Running Deployment Readiness Check...")
        results['deployment_readiness'] = readiness_checker.run_comprehensive_check()
        
        # 2. Run test suite
        print("\n🧪 Running Test Suite...")
        results['test_results'] = test_suite.run_all_tests()
        
        # 3. Run security audit
        print("\n🔒 Running Security Audit...")
        results['security_audit'] = security_auditor.run_comprehensive_audit()
        
        # 4. Run performance optimization
        print("\n⚡ Running Performance Optimization...")
        results['performance'] = performance_optimizer.optimize_application()
        
        # Generate final report
        generate_final_report(results)
        
        print("\n✅ Complete deployment check finished successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Deployment check failed: {str(e)}")
        return False

def generate_final_report(results):
    """Generate final deployment report"""
    report_path = project_root / 'FINAL_DEPLOYMENT_REPORT.md'
    
    report_content = f"""# Final Deployment Report

## Overview
This report summarizes the complete deployment readiness assessment for the Mental Health App.

## Summary Statistics
"""
    
    # Calculate overall statistics
    total_checks = 0
    passed_checks = 0
    
    for category, category_results in results.items():
        if isinstance(category_results, dict):
            report_content += f"\n### {category.replace('_', ' ').title()}\n"
            
            # Count checks in this category
            category_total = 0
            category_passed = 0
            
            def count_checks(data, prefix=""):
                nonlocal category_total, category_passed
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict) and 'status' in value:
                            category_total += 1
                            if value['status'] == 'pass':
                                category_passed += 1
                            status_emoji = '✅' if value['status'] == 'pass' else '⚠️' if value['status'] == 'warning' else '❌'
                            report_content += f"- {status_emoji} {prefix}{key.replace('_', ' ').title()}: {value.get('message', 'No message')}\n"
                        elif isinstance(value, dict):
                            count_checks(value, f"{prefix}{key.replace('_', ' ').title()} - ")
            
            count_checks(category_results)
            
            total_checks += category_total
            passed_checks += category_passed
            
            if category_total > 0:
                success_rate = (category_passed / category_total) * 100
                report_content += f"\n**Category Success Rate: {success_rate:.1f}% ({category_passed}/{category_total})**\n"
    
    # Overall statistics
    if total_checks > 0:
        overall_success_rate = (passed_checks / total_checks) * 100
        report_content += f"""
## Overall Assessment

- **Total Checks**: {total_checks}
- **Passed Checks**: {passed_checks}
- **Overall Success Rate**: {overall_success_rate:.1f}%

"""
        
        if overall_success_rate >= 90:
            report_content += "🚀 **DEPLOYMENT APPROVED** - Application is ready for production deployment!\n"
        elif overall_success_rate >= 75:
            report_content += "⚠️ **CONDITIONAL APPROVAL** - Address warnings before deployment.\n"
        else:
            report_content += "❌ **DEPLOYMENT NOT APPROVED** - Critical issues must be resolved.\n"
    
    report_content += f"""
## Deployment Checklist

### Pre-Deployment
- [ ] All critical issues resolved
- [ ] Security audit passed
- [ ] Performance benchmarks met
- [ ] Test coverage adequate
- [ ] Documentation complete

### Deployment
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] Monitoring systems active
- [ ] Rollback procedures tested
- [ ] Team notified of deployment

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] User acceptance testing
- [ ] Performance monitoring active
- [ ] Incident response ready

---
*Report generated: {import time; time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"\n📊 Final deployment report generated: {report_path}")

if __name__ == "__main__":
    success = run_complete_deployment_check()
    sys.exit(0 if success else 1)