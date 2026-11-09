// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title TradingLeaderboard
 * @dev Manages the leaderboard for the Decentralized Algorithmic Trading Arena.
 * Only the authorized backend wallet (the owner) can submit scores verified by the off-chain engine.
 */
contract TradingLeaderboard is Ownable {

    // --- Structs ---

    struct Submission {
        address walletAddress;    // Wallet address of the trader/algorithm owner
        string algorithmName;     // Name/identifier of the submitted algorithm
        int256 pnlPercentage;     // PnL percentage scaled by 100 (e.g., 1234 represents 12.34%)
        uint256 timestamp;        // Timestamp when the submission was recorded on-chain
    }

    // --- State Variables ---

    // Array storing all submissions chronologically
    Submission[] public submissions;

    // Mapping from a user's wallet address to an array of indices in the submissions array
    // This facilitates efficient off-chain querying of a user's historical submissions
    mapping(address => uint256[]) private userSubmissionIndices;

    // --- Events ---

    /**
     * @dev Emitted when a new score is successfully submitted and verified on-chain.
     * @param walletAddress The trader's wallet address.
     * @param algorithmName The name of the algorithm.
     * @param pnlPercentage The algorithm's PnL percentage (scaled by 100).
     * @param timestamp The block timestamp of the submission.
     */
    event ScoreSubmitted(
        address indexed walletAddress,
        string algorithmName,
        int256 pnlPercentage,
        uint256 timestamp
    );

    // --- Constructor ---

    /**
     * @dev Initializes the contract setting the deployer (or a designated address) as the initial owner.
     * @param initialOwner The address authorized as the initial owner (e.g., the backend wallet).
     */
    constructor(address initialOwner) Ownable(initialOwner) {}

    // --- External/Public Write Functions ---

    /**
     * @notice Submits a new algorithm verification score. Only callable by the contract owner.
     * @dev Appends a new Submission struct to the array and updates user indexing mappings.
     * @param userWallet The wallet address of the trader.
     * @param algoName The name of the algorithm.
     * @param pnl The verified PnL percentage, scaled by 100.
     */
    function submitScore(
        address userWallet,
        string memory algoName,
        int256 pnl
    ) public onlyOwner {
        require(userWallet != address(0), "TradingLeaderboard: Invalid user wallet address");
        require(bytes(algoName).length > 0, "TradingLeaderboard: Algorithm name cannot be empty");

        uint256 currentTimestamp = block.timestamp;

        // Construct the new Submission struct
        Submission memory newSubmission = Submission({
            walletAddress: userWallet,
            algorithmName: algoName,
            pnlPercentage: pnl,
            timestamp: currentTimestamp
        });

        // Store the submission
        submissions.push(newSubmission);

        // Map the submission index to the user's wallet for easy querying
        uint256 newSubmissionIndex = submissions.length - 1;
        userSubmissionIndices[userWallet].push(newSubmissionIndex);

        // Emit the score submission event
        emit ScoreSubmitted(userWallet, algoName, pnl, currentTimestamp);
    }

    // --- External/Public Read Functions ---

    /**
     * @notice Returns the total number of submissions.
     * @return The length of the submissions array.
     */
    function getTotalSubmissions() external view returns (uint256) {
        return submissions.length;
    }

    /**
     * @notice Returns all submission indices for a specific user wallet.
     * @param userWallet The wallet address of the trader.
     * @return An array of indices corresponding to the user's submissions in the main array.
     */
    function getUserSubmissionIndices(address userWallet) external view returns (uint256[] memory) {
        return userSubmissionIndices[userWallet];
    }

    /**
     * @notice Retrieves submissions within a specified range (for pagination).
     * @dev Extremely gas efficient for front-ends to fetch leaderboard data in chunks.
     * @param offset The starting index.
     * @param limit The maximum number of submissions to return.
     * @return An array of Submission structs.
     */
    function getSubmissionsRange(
        uint256 offset,
        uint256 limit
    ) external view returns (Submission[] memory) {
        uint256 total = submissions.length;
        if (offset >= total || limit == 0) {
            return new Submission[](0);
        }

        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }

        uint256 resultSize = end - offset;
        Submission[] memory rangeSubmissions = new Submission[](resultSize);

        for (uint256 i = 0; i < resultSize; i++) {
            rangeSubmissions[i] = submissions[offset + i];
        }

        return rangeSubmissions;
    }
}
