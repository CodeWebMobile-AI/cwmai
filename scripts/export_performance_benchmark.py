"""
Export Performance Benchmark Script

Comprehensive benchmarking tool for the CWMAI data export system.
Tests performance across all data types and export formats with detailed metrics.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_export_service import DataExportService, DataType, ExportFormat


class ExportBenchmark:
    """Comprehensive benchmarking for export functionality."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize the benchmark tool.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        self.export_service = DataExportService(output_dir=os.path.join(output_dir, "exports"))
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Benchmark results storage
        self.results = []
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarks across all data types and formats.
        
        Returns:
            Dictionary containing all benchmark results
        """
        print("Starting comprehensive export performance benchmark...")
        print("=" * 60)
        
        benchmark_start = time.time()
        
        # Test matrix: all data types x all export formats
        test_matrix = []
        for data_type in DataType:
            for export_format in ExportFormat:
                test_matrix.append((data_type, export_format))
        
        print(f"Running {len(test_matrix)} benchmark tests...")
        
        for i, (data_type, export_format) in enumerate(test_matrix, 1):
            print(f"\n[{i}/{len(test_matrix)}] Testing {data_type.value} -> {export_format.value}")
            
            try:
                # Run benchmark
                result = self.export_service.get_export_performance_benchmark(
                    data_type, export_format
                )
                
                # Add additional metadata
                result.update({
                    "test_number": i,
                    "data_type_enum": data_type.value,
                    "format_enum": export_format.value
                })
                
                self.results.append(result)
                
                # Print immediate results
                if result["success"]:
                    print(f"  ✅ Success: {result['execution_time_seconds']}s, "
                          f"{result['file_size_bytes']} bytes, "
                          f"{result['memory_used_mb']} MB")
                else:
                    print(f"  ❌ Failed: {result['output_file']}")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                error_result = {
                    "test_number": i,
                    "data_type": data_type.value,
                    "export_format": export_format.value,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.results.append(error_result)
        
        benchmark_end = time.time()
        total_time = benchmark_end - benchmark_start
        
        # Generate summary
        summary = self._generate_summary(total_time)
        
        # Save results
        self._save_results(summary)
        
        print("\n" + "=" * 60)
        print("Benchmark completed!")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        return summary
    
    def run_format_comparison(self, data_type: DataType) -> Dict[str, Any]:
        """Run performance comparison across formats for a specific data type.
        
        Args:
            data_type: Data type to benchmark
            
        Returns:
            Comparison results
        """
        print(f"Running format comparison for {data_type.value}...")
        
        comparison_results = []
        
        for export_format in ExportFormat:
            print(f"  Testing {export_format.value}...")
            
            # Run multiple iterations for more accurate results
            iterations = 3
            iteration_results = []
            
            for i in range(iterations):
                try:
                    result = self.export_service.get_export_performance_benchmark(
                        data_type, export_format
                    )
                    if result["success"]:
                        iteration_results.append(result)
                except Exception as e:
                    print(f"    Iteration {i+1} failed: {str(e)}")
            
            if iteration_results:
                # Calculate averages
                avg_time = sum(r["execution_time_seconds"] for r in iteration_results) / len(iteration_results)
                avg_memory = sum(r["memory_used_mb"] for r in iteration_results) / len(iteration_results)
                avg_size = sum(r["file_size_bytes"] for r in iteration_results) / len(iteration_results)
                
                comparison_results.append({
                    "format": export_format.value,
                    "avg_execution_time": round(avg_time, 3),
                    "avg_memory_usage": round(avg_memory, 2),
                    "avg_file_size": round(avg_size, 0),
                    "iterations": len(iteration_results),
                    "success_rate": len(iteration_results) / iterations
                })
        
        # Sort by execution time
        comparison_results.sort(key=lambda x: x["avg_execution_time"])
        
        comparison = {
            "data_type": data_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "comparison_results": comparison_results,
            "fastest_format": comparison_results[0]["format"] if comparison_results else None,
            "summary": self._generate_format_comparison_summary(comparison_results)
        }
        
        return comparison
    
    def run_scalability_test(self, data_type: DataType, export_format: ExportFormat) -> Dict[str, Any]:
        """Run scalability test with varying data sizes.
        
        Args:
            data_type: Data type to test
            export_format: Export format to test
            
        Returns:
            Scalability test results
        """
        print(f"Running scalability test for {data_type.value} -> {export_format.value}...")
        
        # Note: This is a simplified scalability test
        # In a real scenario, you'd want to generate datasets of varying sizes
        scalability_results = []
        
        # Run benchmark multiple times to simulate varying loads
        for load_factor in [1, 2, 3]:
            print(f"  Testing load factor {load_factor}...")
            
            start_time = time.time()
            result = self.export_service.get_export_performance_benchmark(
                data_type, export_format
            )
            end_time = time.time()
            
            if result["success"]:
                scalability_results.append({
                    "load_factor": load_factor,
                    "execution_time": result["execution_time_seconds"],
                    "memory_usage": result["memory_used_mb"],
                    "file_size": result["file_size_bytes"]
                })
        
        return {
            "data_type": data_type.value,
            "export_format": export_format.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scalability_results": scalability_results,
            "linear_performance": self._analyze_linearity(scalability_results)
        }
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary.
        
        Args:
            total_time: Total benchmark execution time
            
        Returns:
            Summary dictionary
        """
        successful_tests = [r for r in self.results if r.get("success", False)]
        failed_tests = [r for r in self.results if not r.get("success", False)]
        
        # Performance statistics
        if successful_tests:
            execution_times = [r["execution_time_seconds"] for r in successful_tests]
            memory_usage = [r["memory_used_mb"] for r in successful_tests]
            file_sizes = [r["file_size_bytes"] for r in successful_tests]
            
            performance_stats = {
                "avg_execution_time": round(sum(execution_times) / len(execution_times), 3),
                "min_execution_time": round(min(execution_times), 3),
                "max_execution_time": round(max(execution_times), 3),
                "avg_memory_usage": round(sum(memory_usage) / len(memory_usage), 2),
                "avg_file_size": round(sum(file_sizes) / len(file_sizes), 0)
            }
        else:
            performance_stats = {}
        
        # Format analysis
        format_performance = {}
        for format_type in ExportFormat:
            format_results = [r for r in successful_tests if r.get("export_format") == format_type.value]
            if format_results:
                avg_time = sum(r["execution_time_seconds"] for r in format_results) / len(format_results)
                format_performance[format_type.value] = {
                    "count": len(format_results),
                    "avg_time": round(avg_time, 3),
                    "success_rate": len(format_results) / len([r for r in self.results if r.get("export_format") == format_type.value])
                }
        
        # Data type analysis
        datatype_performance = {}
        for data_type in DataType:
            datatype_results = [r for r in successful_tests if r.get("data_type") == data_type.value]
            if datatype_results:
                avg_time = sum(r["execution_time_seconds"] for r in datatype_results) / len(datatype_results)
                datatype_performance[data_type.value] = {
                    "count": len(datatype_results),
                    "avg_time": round(avg_time, 3),
                    "success_rate": len(datatype_results) / len([r for r in self.results if r.get("data_type") == data_type.value])
                }
        
        return {
            "benchmark_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_execution_time": round(total_time, 2),
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.results) if self.results else 0
            },
            "performance_statistics": performance_stats,
            "format_performance": format_performance,
            "datatype_performance": datatype_performance,
            "recommendations": self._generate_recommendations(format_performance, datatype_performance),
            "detailed_results": self.results
        }
    
    def _generate_format_comparison_summary(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary for format comparison.
        
        Args:
            comparison_results: List of format comparison results
            
        Returns:
            Summary dictionary
        """
        if not comparison_results:
            return {}
        
        fastest = comparison_results[0]
        slowest = comparison_results[-1]
        
        return {
            "fastest_format": fastest["format"],
            "slowest_format": slowest["format"],
            "speed_difference": round(slowest["avg_execution_time"] / fastest["avg_execution_time"], 2),
            "memory_comparison": {
                "most_efficient": min(comparison_results, key=lambda x: x["avg_memory_usage"])["format"],
                "least_efficient": max(comparison_results, key=lambda x: x["avg_memory_usage"])["format"]
            },
            "file_size_comparison": {
                "smallest": min(comparison_results, key=lambda x: x["avg_file_size"])["format"],
                "largest": max(comparison_results, key=lambda x: x["avg_file_size"])["format"]
            }
        }
    
    def _analyze_linearity(self, scalability_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if performance scales linearly.
        
        Args:
            scalability_results: List of scalability test results
            
        Returns:
            Linearity analysis
        """
        if len(scalability_results) < 2:
            return {"analysis": "insufficient_data"}
        
        # Simple linearity check based on execution time growth
        time_ratios = []
        for i in range(1, len(scalability_results)):
            ratio = scalability_results[i]["execution_time"] / scalability_results[i-1]["execution_time"]
            time_ratios.append(ratio)
        
        avg_ratio = sum(time_ratios) / len(time_ratios)
        
        return {
            "avg_growth_ratio": round(avg_ratio, 2),
            "is_linear": 0.8 <= avg_ratio <= 1.2,  # Within 20% of linear growth
            "performance_trend": "linear" if 0.8 <= avg_ratio <= 1.2 else ("sublinear" if avg_ratio < 0.8 else "superlinear")
        }
    
    def _generate_recommendations(self, format_performance: Dict[str, Any], datatype_performance: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations.
        
        Args:
            format_performance: Format performance data
            datatype_performance: Data type performance data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Format recommendations
        if format_performance:
            fastest_format = min(format_performance.items(), key=lambda x: x[1]["avg_time"])
            recommendations.append(f"For best performance, use {fastest_format[0]} format (avg: {fastest_format[1]['avg_time']}s)")
            
            # Find most reliable format
            most_reliable = max(format_performance.items(), key=lambda x: x[1]["success_rate"])
            if most_reliable[1]["success_rate"] < 1.0:
                recommendations.append(f"For reliability, use {most_reliable[0]} format ({most_reliable[1]['success_rate']:.1%} success rate)")
        
        # Data type recommendations
        if datatype_performance:
            fastest_datatype = min(datatype_performance.items(), key=lambda x: x[1]["avg_time"])
            slowest_datatype = max(datatype_performance.items(), key=lambda x: x[1]["avg_time"])
            
            if fastest_datatype[1]["avg_time"] * 2 < slowest_datatype[1]["avg_time"]:
                recommendations.append(f"Note: {slowest_datatype[0]} exports take significantly longer than {fastest_datatype[0]}")
        
        # General recommendations
        recommendations.extend([
            "Use JSON format for programmatic processing and data interchange",
            "Use CSV format for data analysis and spreadsheet compatibility",
            "Use PDF format for reports and documentation",
            "Consider filtering data to reduce export time for large datasets",
            "Monitor memory usage for large exports in production environments"
        ])
        
        return recommendations
    
    def _save_results(self, summary: Dict[str, Any]) -> None:
        """Save benchmark results to files.
        
        Args:
            summary: Benchmark summary data
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save summary report as text
        report_file = os.path.join(self.output_dir, f"benchmark_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("CWMAI Data Export Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            metadata = summary["benchmark_metadata"]
            f.write(f"Benchmark Date: {metadata['timestamp']}\n")
            f.write(f"Total Execution Time: {metadata['total_execution_time']} seconds\n")
            f.write(f"Tests Run: {metadata['total_tests']}\n")
            f.write(f"Success Rate: {metadata['success_rate']:.1%}\n\n")
            
            if summary.get("performance_statistics"):
                stats = summary["performance_statistics"]
                f.write("Performance Statistics:\n")
                f.write(f"  Average Execution Time: {stats['avg_execution_time']} seconds\n")
                f.write(f"  Range: {stats['min_execution_time']} - {stats['max_execution_time']} seconds\n")
                f.write(f"  Average Memory Usage: {stats['avg_memory_usage']} MB\n")
                f.write(f"  Average File Size: {stats['avg_file_size']} bytes\n\n")
            
            f.write("Recommendations:\n")
            for i, rec in enumerate(summary.get("recommendations", []), 1):
                f.write(f"  {i}. {rec}\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Summary report saved to: {report_file}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="CWMAI Export Performance Benchmark")
    parser.add_argument("--mode", choices=["comprehensive", "format", "scalability"], 
                       default="comprehensive", help="Benchmark mode")
    parser.add_argument("--data-type", choices=[dt.value for dt in DataType], 
                       help="Data type for format/scalability tests")
    parser.add_argument("--format", choices=[ef.value for ef in ExportFormat], 
                       help="Format for scalability tests")
    parser.add_argument("--output-dir", default="benchmark_results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create benchmark tool
    benchmark = ExportBenchmark(output_dir=args.output_dir)
    
    try:
        if args.mode == "comprehensive":
            results = benchmark.run_comprehensive_benchmark()
            
        elif args.mode == "format":
            if not args.data_type:
                print("Error: --data-type required for format comparison")
                return 1
            
            data_type = DataType(args.data_type)
            results = benchmark.run_format_comparison(data_type)
            
            print(f"\nFormat Comparison Results for {data_type.value}:")
            for result in results["comparison_results"]:
                print(f"  {result['format']}: {result['avg_execution_time']}s "
                      f"({result['avg_memory_usage']} MB, {result['avg_file_size']} bytes)")
            
        elif args.mode == "scalability":
            if not args.data_type or not args.format:
                print("Error: --data-type and --format required for scalability test")
                return 1
            
            data_type = DataType(args.data_type)
            export_format = ExportFormat(args.format)
            results = benchmark.run_scalability_test(data_type, export_format)
            
            print(f"\nScalability Test Results:")
            print(f"Performance Trend: {results['linear_performance']['performance_trend']}")
            
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())