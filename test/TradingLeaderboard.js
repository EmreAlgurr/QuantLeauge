const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("TradingLeaderboard", function () {
  let TradingLeaderboard;
  let leaderboard;
  let owner;
  let backendWallet;
  let trader1;
  let trader2;

  beforeEach(async function () {
    // Get signers
    [owner, backendWallet, trader1, trader2] = await ethers.getSigners();

    // Deploy contract setting backendWallet as owner
    TradingLeaderboard = await ethers.getContractFactory("TradingLeaderboard");
    leaderboard = await TradingLeaderboard.deploy(backendWallet.address);
    await leaderboard.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await leaderboard.owner()).to.equal(backendWallet.address);
    });

    it("Should start with 0 submissions", async function () {
      expect(await leaderboard.getTotalSubmissions()).to.equal(0);
    });
  });

  describe("Score Submission", function () {
    it("Should allow the owner (backend) to submit a score", async function () {
      const algoName = "MACD-Cross-v1";
      const pnl = 1550; // 15.50%

      // Submit score
      const tx = await leaderboard.connect(backendWallet).submitScore(trader1.address, algoName, pnl);
      await tx.wait();

      expect(await leaderboard.getTotalSubmissions()).to.equal(1);

      // Verify the details in the submissions array
      const submission = await leaderboard.submissions(0);
      expect(submission.walletAddress).to.equal(trader1.address);
      expect(submission.algorithmName).to.equal(algoName);
      expect(submission.pnlPercentage).to.equal(pnl);
      expect(submission.timestamp).to.be.gt(0);
    });

    it("Should reject score submissions from non-owners", async function () {
      const algoName = "RSI-Trend";
      const pnl = -500; // -5.00%

      // trader1 tries to call submitScore, should fail because trader1 is not owner
      // In OpenZeppelin v5, this reverts with OwnableUnauthorizedAccount(address account)
      await expect(
        leaderboard.connect(trader1).submitScore(trader2.address, algoName, pnl)
      ).to.be.revertedWithCustomError(leaderboard, "OwnableUnauthorizedAccount")
       .withArgs(trader1.address);
    });

    it("Should reject invalid user wallet or empty algorithm name", async function () {
      // Zero address
      await expect(
        leaderboard.connect(backendWallet).submitScore(ethers.ZeroAddress, "Test", 100)
      ).to.be.revertedWith("TradingLeaderboard: Invalid user wallet address");

      // Empty algo name
      await expect(
        leaderboard.connect(backendWallet).submitScore(trader1.address, "", 100)
      ).to.be.revertedWith("TradingLeaderboard: Algorithm name cannot be empty");
    });
  });

  describe("Queries and Pagination", function () {
    beforeEach(async function () {
      // Submit multiple scores
      await leaderboard.connect(backendWallet).submitScore(trader1.address, "Algo1", 1000);
      await leaderboard.connect(backendWallet).submitScore(trader2.address, "Algo2", 2000);
      await leaderboard.connect(backendWallet).submitScore(trader1.address, "Algo3", -500);
    });

    it("Should track user submissions correctly", async function () {
      const indicesTrader1 = await leaderboard.getUserSubmissionIndices(trader1.address);
      expect(indicesTrader1.length).to.equal(2);
      expect(indicesTrader1[0]).to.equal(0n);
      expect(indicesTrader1[1]).to.equal(2n);

      const indicesTrader2 = await leaderboard.getUserSubmissionIndices(trader2.address);
      expect(indicesTrader2.length).to.equal(1);
      expect(indicesTrader2[0]).to.equal(1n);
    });

    it("Should paginate submissions correctly", async function () {
      // Get all 3 submissions
      let range = await leaderboard.getSubmissionsRange(0, 3);
      expect(range.length).to.equal(3);
      expect(range[0].algorithmName).to.equal("Algo1");
      expect(range[1].algorithmName).to.equal("Algo2");
      expect(range[2].algorithmName).to.equal("Algo3");

      // Offset 1, limit 2 (should return Algo2, Algo3)
      range = await leaderboard.getSubmissionsRange(1, 2);
      expect(range.length).to.equal(2);
      expect(range[0].algorithmName).to.equal("Algo2");
      expect(range[1].algorithmName).to.equal("Algo3");

      // Out of bounds offset should return an empty array
      range = await leaderboard.getSubmissionsRange(5, 2);
      expect(range.length).to.equal(0);
    });
  });
});
