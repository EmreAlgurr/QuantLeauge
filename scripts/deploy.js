const hre = require("hardhat");

async function main() {
  console.log("=================================================");
  console.log("Starting deployment of TradingLeaderboard...");
  console.log("=================================================");

  // Get signers
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contract with account:", deployer.address);
  
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH");

  // Determine initial owner: use BACKEND_OWNER_WALLET if set, otherwise use deployer
  let initialOwner = process.env.BACKEND_OWNER_WALLET;
  if (!initialOwner || initialOwner.trim() === "" || initialOwner === "0x0000000000000000000000000000000000000000") {
    initialOwner = deployer.address;
    console.log("No valid BACKEND_OWNER_WALLET found in .env. Defaulting to deployer address.");
  }
  
  console.log("Setting initial contract owner to:", initialOwner);

  // Get the ContractFactory for TradingLeaderboard
  const TradingLeaderboard = await hre.ethers.getContractFactory("TradingLeaderboard");

  // Deploy the contract with the initial owner as constructor argument
  const leaderboard = await TradingLeaderboard.deploy(initialOwner);

  console.log("Waiting for deployment confirmation...");
  await leaderboard.waitForDeployment();

  const contractAddress = await leaderboard.getAddress();
  const txHash = leaderboard.deploymentTransaction().hash;

  console.log("\n=================================================");
  console.log("Contract successfully deployed!");
  console.log("Contract Address:", contractAddress);
  console.log("Transaction Hash:", txHash);
  console.log("Sepolia Explorer: https://sepolia.etherscan.io/address/" + contractAddress);
  console.log("=================================================");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\nDeployment failed with error:", error);
    process.exit(1);
  });
