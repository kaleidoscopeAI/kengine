import asyncio
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from kaleidoscope_core import KaleidoscopeCore
from kaleidoscope_core.types import SimulationConfig

# Configure logging with more detailed levels 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stress_test.log"),
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "stress_test_detailed.log",
            maxBytes=10_000_000,
            backupCount=5
        )
    ]
)
logger = logging.getLogger("KaleidoscopeStressTest")

async def stress_test(concurrent_jobs=10, duration_seconds=60, fail_fast=True):
    """Run stress test with specified parameters."""
    logger.info(f"Starting stress test with {concurrent_jobs} concurrent jobs for {duration_seconds}s")
    
    error_count = 0
    max_errors = 5 if fail_fast else float('inf')
    
    try:
        async with KaleidoscopeCore() as core:
            tasks = []
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < duration_seconds:
                # Clean up completed tasks first
                tasks = [t for t in tasks if not t.done()]
                
                while len(tasks) < concurrent_jobs:
                    config = SimulationConfig(
                        timeout=30,  # Add timeout
                        retries=3    # Add retry count
                    )
                    task = asyncio.create_task(core.run_simulation(config))
                    tasks.append(task)
                
                try:
                    # Wait for any task to complete with timeout
                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=5.0
                    )
                    
                    if not done:
                        logger.warning("No tasks completed in timeout window")
                        continue
                        
                    # Process completed tasks
                    for task in done:
                        try:
                            result = await task
                            logger.info(f"Task completed successfully: {result}")
                        except Exception as e:
                            error_count += 1
                            logger.error(f"Task failed: {str(e)}")
                            if error_count >= max_errors:
                                raise RuntimeError(f"Too many errors ({error_count})")
                        finally:
                            tasks.remove(task)
                            
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for tasks")
                
            # Final cleanup - cancel any remaining tasks
            remaining_count = len(tasks)
            if remaining_count:
                logger.info(f"Cancelling {remaining_count} remaining tasks...")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for cancellations
                await asyncio.gather(*tasks, return_exceptions=True)
    
    except Exception as e:
        logger.error(f"Stress test failed: {str(e)}")
        raise
    finally:
        logger.info(f"Stress test completed. Total errors: {error_count}")

async def main():
    await stress_test(concurrent_jobs=20, duration_seconds=120)

if __name__ == "__main__":
    asyncio.run(main())
