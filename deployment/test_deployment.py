#!/usr/bin/env python3
"""
Deployment Testing Script
========================

Comprehensive testing suite for the production deployment.
"""

import asyncio
import json
import time
from typing import Dict, Any, List
import httpx
import structlog

logger = structlog.get_logger(__name__)


class DeploymentTester:
    """
    Comprehensive deployment testing suite.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _record_result(self, test_name: str, success: bool, duration: float, details: Dict = None):
        """Record test result."""
        result = {
            "test_name": test_name,
            "success": success,
            "duration": duration,
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✓" if success else "✗"
        print(f"{status} {test_name} ({duration:.3f}s)")
        
        if not success and details:
            print(f"  Error: {details.get('error', 'Unknown error')}")
    
    async def test_health_check(self) -> bool:
        """Test basic health check."""
        start_time = time.time()
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                success = data.get("status") == "healthy"
                self._record_result("Health Check", success, duration, data)
                return success
            else:
                self._record_result("Health Check", False, duration, 
                                  {"error": f"HTTP {response.status_code}"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Health Check", False, duration, {"error": str(e)})
            return False
    
    async def test_readiness_check(self) -> bool:
        """Test readiness probe."""
        start_time = time.time()
        
        try:
            response = await self.client.get(f"{self.base_url}/ready")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            details = {"status_code": response.status_code}
            
            if success:
                data = response.json()
                details["response"] = data
            
            self._record_result("Readiness Check", success, duration, details)
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Readiness Check", False, duration, {"error": str(e)})
            return False
    
    async def test_text_generation(self) -> bool:
        """Test text generation endpoint."""
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        payload = {
            "prompt": "ሰላም እንዴት ነሽ",
            "max_length": 30,
            "temperature": 1.0,
            "enable_cultural_safety": True
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=headers
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["generated_text", "generation_stats", "cultural_safety", "performance_metrics"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self._record_result("Text Generation", False, duration,
                                      {"error": f"Missing fields: {missing_fields}"})
                    return False
                
                # Check if response time meets target
                meets_sla = duration < 0.2  # 200ms target
                
                details = {
                    "generated_text_length": len(data["generated_text"]),
                    "meets_sla": meets_sla,
                    "cultural_safety_passed": data["cultural_safety"].get("passed", False),
                    "inference_time": data["performance_metrics"].get("inference_duration", 0)
                }
                
                self._record_result("Text Generation", True, duration, details)
                return True
                
            else:
                error_detail = response.text if response.status_code != 422 else response.json()
                self._record_result("Text Generation", False, duration,
                                  {"error": f"HTTP {response.status_code}", "detail": error_detail})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Text Generation", False, duration, {"error": str(e)})
            return False
    
    async def test_cultural_safety(self) -> bool:
        """Test cultural safety validation."""
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        # Test with potentially problematic content
        payload = {
            "prompt": "ቡና is addictive and dangerous",
            "max_length": 20,
            "enable_cultural_safety": True
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=headers
            )
            duration = time.time() - start_time
            
            # This should either succeed with warnings or fail with cultural safety error
            if response.status_code in [200, 400]:
                if response.status_code == 400:
                    # Expected cultural safety rejection
                    error_data = response.json()
                    if "cultural safety" in error_data.get("detail", "").lower():
                        self._record_result("Cultural Safety", True, duration,
                                          {"result": "Correctly blocked unsafe content"})
                        return True
                else:
                    # Check if cultural safety warnings are present
                    data = response.json()
                    cultural_safety = data.get("cultural_safety", {})
                    
                    if not cultural_safety.get("passed", True):
                        self._record_result("Cultural Safety", True, duration,
                                          {"result": "Generated with safety warnings"})
                        return True
                
                self._record_result("Cultural Safety", False, duration,
                                  {"error": "Did not detect cultural safety issues"})
                return False
            else:
                self._record_result("Cultural Safety", False, duration,
                                  {"error": f"HTTP {response.status_code}"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Cultural Safety", False, duration, {"error": str(e)})
            return False
    
    async def test_model_info(self) -> bool:
        """Test model information endpoint."""
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        try:
            response = await self.client.get(f"{self.base_url}/model/info", headers=headers)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                required_fields = ["architecture", "performance", "capabilities"]
                has_required = all(field in data for field in required_fields)
                
                details = {
                    "has_required_fields": has_required,
                    "model_type": data.get("architecture", {}).get("type"),
                    "total_parameters": data.get("architecture", {}).get("total_parameters")
                }
                
                self._record_result("Model Info", has_required, duration, details)
                return has_required
            else:
                self._record_result("Model Info", False, duration,
                                  {"error": f"HTTP {response.status_code}"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Model Info", False, duration, {"error": str(e)})
            return False
    
    async def test_metrics_endpoint(self) -> bool:
        """Test Prometheus metrics endpoint."""
        start_time = time.time()
        
        try:
            response = await self.client.get(f"{self.base_url}/metrics")
            duration = time.time() - start_time
            
            if response.status_code == 200:
                metrics_text = response.text
                
                # Check for key metrics
                expected_metrics = [
                    "http_requests_total",
                    "model_inference_duration_seconds",
                    "cultural_safety_checks_total",
                    "system_cpu_usage_percent"
                ]
                
                missing_metrics = [metric for metric in expected_metrics 
                                 if metric not in metrics_text]
                
                success = len(missing_metrics) == 0
                details = {
                    "metrics_found": len(expected_metrics) - len(missing_metrics),
                    "missing_metrics": missing_metrics
                }
                
                self._record_result("Metrics Endpoint", success, duration, details)
                return success
            else:
                self._record_result("Metrics Endpoint", False, duration,
                                  {"error": f"HTTP {response.status_code}"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Metrics Endpoint", False, duration, {"error": str(e)})
            return False
    
    async def test_morpheme_analysis(self) -> bool:
        """Test morpheme analysis endpoint."""
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        payload = {
            "text": "ሰላም እንዴት ነሽ? ቡና ትፈልጊ ዘመን",
            "include_pos_tags": True,
            "include_features": True,
            "include_cultural_context": True
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/analyze-morphemes",
                json=payload,
                headers=headers
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = [
                    "original_text", "word_analyses", "text_complexity", 
                    "dialect_classification", "cultural_safety_score",
                    "linguistic_quality_score", "readability_metrics", "performance_metrics"
                ]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self._record_result("Morpheme Analysis", False, duration,
                                      {"error": f"Missing fields: {missing_fields}"})
                    return False
                
                # Validate word analyses structure
                word_analyses = data.get("word_analyses", [])
                if not word_analyses:
                    self._record_result("Morpheme Analysis", False, duration,
                                      {"error": "No word analyses returned"})
                    return False
                
                # Check for expected fields in word analysis
                first_analysis = word_analyses[0]
                expected_word_fields = ["word", "morphemes", "morpheme_types", "confidence_score"]
                missing_word_fields = [field for field in expected_word_fields if field not in first_analysis]
                
                if missing_word_fields:
                    self._record_result("Morpheme Analysis", False, duration,
                                      {"error": f"Missing word analysis fields: {missing_word_fields}"})
                    return False
                
                details = {
                    "words_analyzed": len(word_analyses),
                    "text_complexity": data.get("text_complexity", 0),
                    "dialect_classification": data.get("dialect_classification"),
                    "cultural_safety_score": data.get("cultural_safety_score", 0),
                    "linguistic_quality_score": data.get("linguistic_quality_score", 0),
                    "analysis_time": data.get("performance_metrics", {}).get("analysis_duration", 0)
                }
                
                self._record_result("Morpheme Analysis", True, duration, details)
                return True
                
            else:
                error_detail = response.text if response.status_code != 422 else response.json()
                self._record_result("Morpheme Analysis", False, duration,
                                  {"error": f"HTTP {response.status_code}", "detail": error_detail})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Morpheme Analysis", False, duration, {"error": str(e)})
            return False

    async def test_batch_generation(self) -> bool:
        """Test batch generation endpoint."""
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        requests_payload = [
            {
                "prompt": "ሰላም",
                "max_length": 10,
                "temperature": 1.0
            },
            {
                "prompt": "እንዴት ነሽ",
                "max_length": 15,
                "temperature": 0.8
            }
        ]
        
        try:
            response = await self.client.post(
                f"{self.base_url}/batch-generate",
                json=requests_payload,
                headers=headers
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if "results" in data and "batch_stats" in data:
                    batch_stats = data["batch_stats"]
                    success = batch_stats.get("successful", 0) > 0
                    
                    details = {
                        "total_requests": batch_stats.get("total_requests", 0),
                        "successful": batch_stats.get("successful", 0),
                        "failed": batch_stats.get("failed", 0),
                        "avg_duration": batch_stats.get("avg_duration_per_request", 0)
                    }
                    
                    self._record_result("Batch Generation", success, duration, details)
                    return success
                else:
                    self._record_result("Batch Generation", False, duration,
                                      {"error": "Invalid response structure"})
                    return False
            else:
                self._record_result("Batch Generation", False, duration,
                                  {"error": f"HTTP {response.status_code}"})
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("Batch Generation", False, duration, {"error": str(e)})
            return False
    
    async def test_load_performance(self, num_requests: int = 10) -> bool:
        """Test load performance with concurrent requests."""
        print(f"Running load test with {num_requests} concurrent requests...")
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        payload = {
            "prompt": "ሰላም",
            "max_length": 20,
            "temperature": 1.0
        }
        
        async def make_request():
            try:
                response = await self.client.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    headers=headers
                )
                return response.status_code == 200, time.time()
            except:
                return False, time.time()
        
        # Create concurrent tasks
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for success, _ in results if success and not isinstance(success, Exception))
        failed = num_requests - successful
        
        # Calculate response times
        response_times = [end_time - start_time for success, end_time in results 
                         if not isinstance(success, Exception)]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Check SLA compliance
        sla_compliant = sum(1 for rt in response_times if rt < 0.2)
        sla_compliance_rate = sla_compliant / len(response_times) * 100 if response_times else 0
        
        details = {
            "total_requests": num_requests,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / num_requests * 100,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "sla_compliance_rate": sla_compliance_rate,
            "requests_per_second": num_requests / total_duration
        }
        
        success = successful > 0 and sla_compliance_rate > 80
        self._record_result("Load Performance", success, total_duration, details)
        
        return success
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all deployment tests."""
        print("🧪 Running Deployment Tests")
        print("=" * 50)
        
        start_time = time.time()
        
        # Core functionality tests
        await self.test_health_check()
        await self.test_readiness_check()
        await self.test_text_generation()
        await self.test_morpheme_analysis()
        await self.test_cultural_safety()
        await self.test_model_info()
        await self.test_metrics_endpoint()
        await self.test_batch_generation()
        
        # Performance tests
        await self.test_load_performance(10)
        
        total_duration = time.time() - start_time
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests * 100,
            "total_duration": total_duration,
            "test_results": self.test_results
        }
        
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test_name']}: {result['details'].get('error', 'Unknown error')}")
        
        return summary


async def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Amharic H-Net deployment")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    async with DeploymentTester(args.url, args.api_key) as tester:
        results = await tester.run_all_tests()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"\nTest results saved to {args.output}")
        
        # Exit with appropriate code
        exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())