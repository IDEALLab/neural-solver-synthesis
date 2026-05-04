# Simulators - Clean Engineering Design Interfaces

This folder contains clean, reusable simulator interfaces for multi-domain engineering design problems. These simulators are designed to be used by reward functions, agents, and any other workflow that needs to evaluate engineering designs.

## 🎯 Available Simulators

- **EPS**: Satellite Electrical Power System design
- **Beams2D**: Structural topology optimization  
- **Knapsack**: Combinatorial optimization

## 🚀 Quick Start

### Basic Usage

```python
from open_r1.simulators import registry

# Single simulation
reward = registry.get_reward("eps", "0123", mission_requirements)

# Get simulation results
results = registry.simulate("beams2d", design_matrix, constraints)

# Batch simulation
rewards = registry.batch_simulate("knapsack", designs, requirements)
```

### Direct Simulator Access

```python
# Get specific simulator
eps_sim = registry.get_simulator("eps")
results = eps_sim.simulate(design, requirements)
capabilities = eps_sim.get_info()
```

## 🤖 Agentic Workflows

### LangGraph Integration

```python
from langgraph import StateGraph
from open_r1.simulators import registry

class DesignState(TypedDict):
    domain: str
    design: Any
    requirements: Dict[str, Any]
    reward: float
    results: Dict[str, float]

def simulate_node(state: DesignState) -> DesignState:
    """Simulate design and update state"""
    domain = state["domain"]
    design = state["design"]
    requirements = state["requirements"]
    
    # Get reward score
    reward = registry.get_reward(domain, design, requirements)
    
    # Get detailed results
    results = registry.simulate(domain, design, requirements)
    
    return {
        **state,
        "reward": reward,
        "results": results
    }

def validate_node(state: DesignState) -> DesignState:
    """Validate design meets requirements"""
    domain = state["domain"]
    design = state["design"]
    requirements = state["requirements"]
    
    simulator = registry.get_simulator(domain)
    is_valid = simulator.validate_design(design, requirements)
    
    if not is_valid:
        return {**state, "reward": 0.0}
    
    return state

# Build LangGraph workflow
workflow = StateGraph(DesignState)
workflow.add_node("validate", validate_node)
workflow.add_node("simulate", simulate_node)
workflow.add_edge("validate", "simulate")
workflow.set_entry_point("validate")

app = workflow.compile()
```

### Model Context Protocol (MCP) Integration

```python
from mcp import Server, StdioServerParameters
from mcp.server.models import InitializationOptions
from open_r1.simulators import registry

class SimulatorMCPServer:
    def __init__(self):
        self.server = Server("simulator-server")
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.server.list_tools()
        async def list_tools():
            return [
                {
                    "name": "simulate_design",
                    "description": "Simulate engineering design",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "domain": {"type": "string", "enum": ["eps", "beams2d", "knapsack"]},
                            "design": {"type": "object"},
                            "requirements": {"type": "object"}
                        },
                        "required": ["domain", "design", "requirements"]
                    }
                },
                {
                    "name": "get_simulator_info",
                    "description": "Get simulator capabilities",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "domain": {"type": "string", "enum": ["eps", "beams2d", "knapsack"]}
                        },
                        "required": ["domain"]
                    }
                }
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "simulate_design":
                domain = arguments["domain"]
                design = arguments["design"]
                requirements = arguments["requirements"]
                
                try:
                    reward = registry.get_reward(domain, design, requirements)
                    results = registry.simulate(domain, design, requirements)
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Simulation complete. Reward: {reward:.4f}, Results: {results}"
                            }
                        ]
                    }
                except Exception as e:
                    return {
                        "content": [
                            {
                                "type": "text", 
                                "text": f"Simulation failed: {str(e)}"
                            }
                        ]
                    }
            
            elif name == "get_simulator_info":
                domain = arguments["domain"]
                try:
                    info = registry.get_simulator_info(domain)
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Simulator info for {domain}: {info}"
                            }
                        ]
                    }
                except Exception as e:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Failed to get simulator info: {str(e)}"
                            }
                        ]
                    }
    
    async def run(self):
        async with StdioServerParameters() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="simulator-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None
                    )
                )
            )

# Run MCP server
if __name__ == "__main__":
    server = SimulatorMCPServer()
    asyncio.run(server.run())
```

## 🔧 Advanced Usage

### Custom Simulator Configuration

```python
# Configure simulator with custom settings
config = {
    "timeout": 30.0,
    "default_requirements": {...}
}

eps_sim = registry.get_simulator("eps", config)
results = eps_sim.simulate(design, requirements)
```

### Batch Processing for Agents

```python
def agent_batch_evaluate(domain, designs, requirements):
    """Batch evaluation for agent workflows"""
    try:
        results = registry.batch_simulate(domain, designs, requirements)
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": []
        }
```

### Error Handling

```python
def safe_simulate(domain, design, requirements):
    """Safe simulation with error handling"""
    try:
        reward = registry.get_reward(domain, design, requirements)
        return {"success": True, "reward": reward}
    except ValueError as e:
        return {"success": False, "error": f"Invalid domain: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Simulation failed: {e}"}
```

## 📋 API Reference

### Registry Methods

- `registry.get_reward(domain, design, requirements)` - Get reward score
- `registry.simulate(domain, design, requirements)` - Get detailed results  
- `registry.get_simulator(domain, config)` - Get simulator instance
- `registry.batch_simulate(domain, designs, requirements)` - Batch simulation
- `registry.list_simulators()` - List available domains
- `registry.get_simulator_info(domain)` - Get simulator capabilities

### Simulator Methods

- `simulator.simulate(design, requirements)` - Run simulation
- `simulator.get_reward(design, requirements)` - Get reward score
- `simulator.validate_design(design, requirements)` - Validate design
- `simulator.get_info()` - Get simulator information

## 🎯 Domain-Specific Usage

### EPS (Satellite Design)
```python
# Design code format: "0123" (orbit, solar_array, battery, dof)
design = "0123"
requirements = {
    "Lifetime": 5,
    "Payload Power": 1000,
    "Delta-V": 500
}
reward = registry.get_reward("eps", design, requirements)
```

### Beams2D (Topology Optimization)
```python
# Design matrix format: numpy array
import numpy as np
design = np.random.rand(14, 28)  # 14x28 design matrix
requirements = {
    "volfrac": 0.4,
    "rmin": 2.0,
    "forcedist": 0.5
}
reward = registry.get_reward("beams2d", design, requirements)
```

### Knapsack (Combinatorial Optimization)
```python
# Design format: list of selected item IDs
design = ["item1", "item3", "item5"]
requirements = {
    "items": [...],  # Available items
    "weight_capacity": 100,
    "volume_capacity": 50
}
reward = registry.get_reward("knapsack", design, requirements)
```

## 🔒 Safety & Security

- **Sandboxed execution**: All code execution is safely sandboxed
- **Input validation**: Designs and requirements are validated
- **Error handling**: Graceful error handling with informative messages
- **Timeout protection**: All simulations have timeout limits

## 🚀 Extending Simulators

### Adding New Domains

```python
# Register new simulator
from .base import BaseSimulator

class NewDomainSimulator(BaseSimulator):
    def _setup(self):
        # Setup logic
        pass
    
    def simulate(self, design, requirements):
        # Simulation logic
        return {"performance": 0.8, "cost": 1000}
    
    def validate_design(self, design, requirements):
        # Validation logic
        return True

# Register with registry
registry.register_simulator("new_domain", NewDomainSimulator)
```

## 📚 Examples

See `agentic_example.py` for a complete agent implementation using these simulators.

## 🤝 Contributing

When adding new simulators:
1. Inherit from `BaseSimulator`
2. Implement required methods
3. Register with `SimulatorRegistry`
4. Add domain-specific documentation
5. Update this README

---

**These simulators are designed to be clean, reusable, and agent-friendly. Use them in any workflow that needs engineering design evaluation!** 🎉
